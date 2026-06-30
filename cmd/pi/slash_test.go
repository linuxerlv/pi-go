package main

import (
	"context"
	"strings"
	"testing"

	"github.com/linuxerlv/pi-go/internal/ai"
	"github.com/linuxerlv/pi-go/internal/harness"
	"github.com/linuxerlv/pi-go/internal/permission"
)

// testProvider is a minimal ai.Provider for cmd/pi tests. StreamSimple emits a
// single complete text turn so harness.Prompt completes, persists a session
// leaf, and writes the jsonl file to disk — which the slash commands that read
// session state (/fork, /name, /export, /sessions, /resume) depend on.
type testProvider struct{}

func (testProvider) ID() string       { return "mock" }
func (testProvider) Name() string     { return "mock" }
func (testProvider) BaseURL() string  { return "" }
func (testProvider) Models() []ai.Model { return nil }
func (testProvider) GetModel(string) (ai.Model, bool) { return ai.Model{}, false }
func (testProvider) StreamSimple(_ context.Context, model ai.Model, _ ai.Context, _ *ai.SimpleStreamOptions) ai.AssistantMessageEventStream {
	s := ai.NewAssistantMessageEventStream()
	go func() {
		msg := ai.AssistantMessage{
			Content:    []ai.ContentBlock{ai.TextContent{Type: "text", Text: "ok"}},
			API:        model.API,
			Provider:   "mock",
			Model:      model.ID,
			StopReason: ai.StopStop,
			Timestamp:  ai.Now(),
		}
		s.Push(ai.StartEvent{Partial: msg})
		s.Push(ai.TextDeltaEvent{ContentIndex: 0, Delta: "ok", Partial: msg})
		s.Push(ai.DoneEvent{Reason: ai.StopStop, Message: msg})
		s.End(nil)
	}()
	return s
}

func newSlashContext() (*strings.Builder, *bool, SlashContext) {
	var sb strings.Builder
	quit := false
	return &sb, &quit, SlashContext{Out: &sb, Quit: &quit}
}

// newTestHarness builds a harness backed by a jsonl session in a temp dir and
// seeds it with one Prompt turn so the session has a leaf and is persisted to
// disk. Returns the harness and its session manager.
func newTestHarness(t *testing.T) (*harness.AgentHarness, *harness.SessionManager) {
	t.Helper()
	mgr := harness.NewSessionManager(t.TempDir())
	sess, err := mgr.Create("h1")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	h := harness.New(harness.Options{
		Provider: testProvider{},
		Model:    ai.Model{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "mock"},
		Session:  sess,
	})
	if _, err := h.Prompt(context.Background(), "seed"); err != nil {
		t.Fatalf("seed prompt: %v", err)
	}
	return h, mgr
}

func TestCommandsRegistered(t *testing.T) {
	cs := commands()
	names := map[string]bool{}
	for _, c := range cs {
		names[c.Name] = true
	}
	for _, want := range []string{"/help", "/quit", "/exit", "/sessions", "/resume", "/new", "/fork", "/name", "/session", "/export", "/import", "/model", "/permission", "/compact", "/tools"} {
		if !names[want] {
			t.Fatalf("missing command %s", want)
		}
	}
	for _, c := range cs {
		if c.Run == nil || c.Description == "" {
			t.Fatalf("command %s incomplete: %+v", c.Name, c)
		}
	}
}

func TestDispatchSlashEmptyLine(t *testing.T) {
	_, _, sc := newSlashContext()
	handled, quit := dispatchSlash("   ", sc)
	if handled || quit {
		t.Fatalf("empty line should not be handled, got handled=%v quit=%v", handled, quit)
	}
}

func TestDispatchSlashUnknown(t *testing.T) {
	sb, _, sc := newSlashContext()
	handled, quit := dispatchSlash("/bogus", sc)
	if !handled || quit {
		t.Fatalf("unknown should be handled but not quit, got %v %v", handled, quit)
	}
	if !strings.Contains(sb.String(), "unknown command") {
		t.Fatalf("expected unknown-command message, got %q", sb.String())
	}
}

func TestDispatchSlashHelp(t *testing.T) {
	sb, _, sc := newSlashContext()
	handled, _ := dispatchSlash("/help", sc)
	if !handled {
		t.Fatal("/help should be handled")
	}
	if !strings.Contains(sb.String(), "commands:") {
		t.Fatalf("expected help output, got %q", sb.String())
	}
}

func TestDispatchSlashQuit(t *testing.T) {
	_, quit, sc := newSlashContext()
	handled, q := dispatchSlash("/quit", sc)
	if !handled || !q {
		t.Fatalf("/quit should set quit, got handled=%v quit=%v", handled, q)
	}
	if !*quit {
		t.Fatal("quit flag should be set true")
	}
}

func TestDispatchSlashExitAlias(t *testing.T) {
	_, quit, sc := newSlashContext()
	_, q := dispatchSlash("/exit", sc)
	if !q || !*quit {
		t.Fatal("/exit should alias to quit")
	}
}

func TestDispatchSlashErrorReported(t *testing.T) {
	sb, _, sc := newSlashContext()
	sc.SessionMgr = nil
	dispatchSlash("/resume", sc)
	if !strings.Contains(sb.String(), "[error:") {
		t.Fatalf("expected error to be reported, got %q", sb.String())
	}
}

func TestCmdModelUsage(t *testing.T) {
	sb, _, sc := newSlashContext()
	if err := cmdModel(nil, sc); err == nil {
		t.Fatal("expected usage error with no args")
	}
	if err := cmdModel([]string{"gpt-4o"}, sc); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(sb.String(), "gpt-4o") {
		t.Fatalf("expected model id in output, got %q", sb.String())
	}
}

func TestCmdCompactAndTools(t *testing.T) {
	sb, _, sc := newSlashContext()
	if err := cmdCompact(nil, sc); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(sb.String(), "compaction") {
		t.Fatalf("expected compaction note, got %q", sb.String())
	}
	sb.Reset()
	if err := cmdTools(nil, sc); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(sb.String(), "read") || !strings.Contains(sb.String(), "glob") {
		t.Fatalf("expected tool list, got %q", sb.String())
	}
}

func TestCmdPermission(t *testing.T) {
	sb, _, sc := newSlashContext()
	if err := cmdPermission(nil, sc); err == nil {
		t.Fatal("expected usage error")
	}
	if err := cmdPermission([]string{"plan"}, sc); err == nil {
		t.Fatal("expected error when permission not enabled")
	}
	c := permission.New(permission.Options{Mode: permission.ModeDefault, Enabled: true})
	sc.Permission = c
	sb.Reset()
	if err := cmdPermission([]string{"acceptEdits"}, sc); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.Mode() != permission.ModeAcceptEdits {
		t.Fatalf("expected mode acceptEdits, got %s", c.Mode())
	}
	if !strings.Contains(sb.String(), "acceptEdits") {
		t.Fatalf("expected mode in output, got %q", sb.String())
	}
}

func TestCmdSessionsEmpty(t *testing.T) {
	sb, _, sc := newSlashContext()
	sc.SessionMgr = harness.NewSessionManager(t.TempDir())
	if err := cmdSessions(nil, sc); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(sb.String(), "no sessions") {
		t.Fatalf("expected empty note, got %q", sb.String())
	}
}

func TestCmdSessionsListsAndMarksCurrent(t *testing.T) {
	_, mgr := newTestHarness(t)
	sb, _, sc := newSlashContext()
	sc.SessionMgr = mgr
	sc.SessionID = "h1"
	if err := cmdSessions(nil, sc); err != nil {
		t.Fatal(err)
	}
	out := sb.String()
	if !strings.Contains(out, "h1") {
		t.Fatalf("expected h1 listed, got %q", out)
	}
	if !strings.Contains(out, "h1 *") {
		t.Fatalf("expected current marker, got %q", out)
	}
}

func TestCmdSessionsNoManager(t *testing.T) {
	_, _, sc := newSlashContext()
	sc.SessionMgr = nil
	if err := cmdSessions(nil, sc); err == nil {
		t.Fatal("expected error when manager is nil")
	}
}

func TestCmdResumeUsageAndOpen(t *testing.T) {
	_, mgr := newTestHarness(t)
	_, _, sc := newSlashContext()
	sc.SessionMgr = mgr
	if err := cmdResume(nil, sc); err == nil {
		t.Fatal("expected usage error")
	}
	if err := cmdResume([]string{"nope"}, sc); err == nil {
		t.Fatal("expected error for missing session")
	}
	sb, _, sc2 := newSlashContext()
	sc2.SessionMgr = mgr
	if err := cmdResume([]string{"h1"}, sc2); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(sb.String(), "h1") {
		t.Fatalf("expected resume message, got %q", sb.String())
	}
}

func TestCmdNewCreates(t *testing.T) {
	mgr := harness.NewSessionManager(t.TempDir())
	sb, _, sc := newSlashContext()
	sc.SessionMgr = mgr
	if err := cmdNew([]string{"work"}, sc); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(sb.String(), "work") {
		t.Fatalf("expected new-session message, got %q", sb.String())
	}
	sb.Reset()
	if err := cmdNew(nil, sc); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(sb.String(), "new session") {
		t.Fatalf("expected default message, got %q", sb.String())
	}
}

func TestCmdSessionInfo(t *testing.T) {
	h, _ := newTestHarness(t)
	sb, _, sc := newSlashContext()
	sc.Harness = h
	if err := cmdSession(nil, sc); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	out := sb.String()
	if !strings.Contains(out, "session: h1") || !strings.Contains(out, "messages:") {
		t.Fatalf("expected session info, got %q", out)
	}
}

func TestCmdName(t *testing.T) {
	h, _ := newTestHarness(t)
	_, _, sc := newSlashContext()
	sc.Harness = h
	if err := cmdName(nil, sc); err == nil {
		t.Fatal("expected usage error")
	}
	if err := cmdName([]string{"my", "label"}, sc); err != nil {
		t.Fatalf("unexpected error setting label: %v", err)
	}
}

func TestCmdFork(t *testing.T) {
	h, _ := newTestHarness(t)
	sb, _, sc := newSlashContext()
	sc.Harness = h
	if err := cmdFork(nil, sc); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(sb.String(), "forked session") {
		t.Fatalf("expected fork message, got %q", sb.String())
	}
	_, _, sc2 := newSlashContext()
	sc2.Harness = nil
	if err := cmdFork(nil, sc2); err == nil {
		t.Fatal("expected error with no harness")
	}
}

func TestCmdExportImportRoundTrip(t *testing.T) {
	dir := t.TempDir()
	h, _ := newTestHarness(t)
	jsonlPath := dir + "/exp.jsonl"
	_, _, sc := newSlashContext()
	sc.Harness = h
	if err := cmdExport([]string{jsonlPath}, sc); err != nil {
		t.Fatalf("export jsonl: %v", err)
	}
	if err := cmdExport(nil, sc); err == nil {
		t.Fatal("expected usage error")
	}
	htmlPath := dir + "/exp.html"
	if err := cmdExport([]string{htmlPath}, sc); err != nil {
		t.Fatalf("export html: %v", err)
	}
	mgr := harness.NewSessionManager(dir)
	sb, _, sc2 := newSlashContext()
	sc2.SessionMgr = mgr
	if err := cmdImport([]string{jsonlPath, "imported-1"}, sc2); err != nil {
		t.Fatalf("import: %v", err)
	}
	if !strings.Contains(sb.String(), "imported") {
		t.Fatalf("expected import message, got %q", sb.String())
	}
	if err := cmdImport(nil, sc2); err == nil {
		t.Fatal("expected usage error")
	}
}

func TestMakeTUISlashHandler(t *testing.T) {
	h, mgr := newTestHarness(t)
	handler := makeTUISlashHandler(h, mgr, "h1", nil, ai.Model{ID: "m"})

	handled, out := handler("/help")
	if !handled || !strings.Contains(out, "commands:") {
		t.Fatalf("expected /help via TUI handler, handled=%v out=%q", handled, out)
	}
	handled2, out2 := handler("/bogus")
	if !handled2 || !strings.Contains(out2, "unknown command") {
		t.Fatalf("expected unknown via TUI handler, got %q", out2)
	}
	handled3, _ := handler("plain text")
	if !handled3 {
		t.Fatal("dispatch should mark unknown as handled")
	}
}
