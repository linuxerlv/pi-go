package harness

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/linuxerlv/pi-go/internal/ai"
)

func TestSessionManagerListCreate(t *testing.T) {
	dir := t.TempDir()
	mgr := NewSessionManager(dir)

	// Create two sessions with messages.
	s1, err := mgr.Create("alpha")
	if err != nil {
		t.Fatal(err)
	}
	s1.AppendMessage(ai.UserMessage{Content: "hi", Timestamp: ai.Now()})
	s1.AppendMessage(ai.AssistantMessage{Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: "hello"}}, Provider: "mock", Model: "m", StopReason: ai.StopStop, Timestamp: ai.Now()})

	s2, err := mgr.Create("beta")
	if err != nil {
		t.Fatal(err)
	}
	s2.AppendMessage(ai.UserMessage{Content: "yo", Timestamp: ai.Now()})

	infos, err := mgr.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(infos) != 2 {
		t.Fatalf("expected 2 sessions, got %d", len(infos))
	}
	// Find alpha and verify message count.
	var alpha *SessionInfo
	for i := range infos {
		if infos[i].ID == "alpha" {
			alpha = &infos[i]
		}
	}
	if alpha == nil {
		t.Fatal("alpha session not found in list")
	}
	if alpha.MessageCount != 2 {
		t.Fatalf("expected alpha to have 2 messages, got %d", alpha.MessageCount)
	}
}

func TestSessionForkBranches(t *testing.T) {
	dir := t.TempDir()
	mgr := NewSessionManager(dir)
	sess, err := mgr.Create("forktest")
	if err != nil {
		t.Fatal(err)
	}
	// Build a base: user + assistant.
	sess.AppendMessage(ai.UserMessage{Content: "base", Timestamp: ai.Now()})
	sess.AppendMessage(ai.AssistantMessage{Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: "base reply"}}, Provider: "mock", Model: "m", StopReason: ai.StopStop, Timestamp: ai.Now()})

	leaf := derefOrEmpty(sess.GetLeafID())
	// Fork from the current leaf.
	newLeaf, err := sess.Fork("")
	if err != nil {
		t.Fatal(err)
	}
	if newLeaf != leaf {
		t.Fatalf("fork should return the same leaf id, got %s want %s", newLeaf, leaf)
	}
	// Append on the new branch.
	sess.AppendMessage(ai.UserMessage{Content: "branch B", Timestamp: ai.Now()})
	ctx := sess.BuildContext()
	// base user + base assistant + branch B user = 3 messages.
	if len(ctx.Messages) != 3 {
		t.Fatalf("expected 3 messages after fork, got %d", len(ctx.Messages))
	}
}

func TestExportHTML(t *testing.T) {
	dir := t.TempDir()
	mgr := NewSessionManager(dir)
	sess, err := mgr.Create("exporthtml")
	if err != nil {
		t.Fatal(err)
	}
	sess.AppendMessage(ai.UserMessage{Content: "hello world", Timestamp: ai.Now()})
	sess.AppendMessage(ai.AssistantMessage{Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: "hi back"}}, Provider: "mock", Model: "m", StopReason: ai.StopStop, Timestamp: ai.Now()})

	out := filepath.Join(dir, "out.html")
	if err := ExportHTML(sess, out); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	if !strings.Contains(s, "<html") || !strings.Contains(s, "hello world") || !strings.Contains(s, "hi back") {
		t.Fatalf("html export missing expected content: %s", s[:200])
	}
}

func TestExportImportJSONL(t *testing.T) {
	dir := t.TempDir()
	mgr := NewSessionManager(dir)
	sess, err := mgr.Create("exportjsonl")
	if err != nil {
		t.Fatal(err)
	}
	sess.AppendMessage(ai.UserMessage{Content: "persist me", Timestamp: ai.Now()})

	out := filepath.Join(dir, "copy.jsonl")
	if err := ExportJSONL(sess, out); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(out); err != nil {
		t.Fatal(err)
	}

	// Import under a new id.
	imported, err := ImportJSONL(mgr, out, "imported")
	if err != nil {
		t.Fatal(err)
	}
	ctx := imported.BuildContext()
	if len(ctx.Messages) != 1 {
		t.Fatalf("expected 1 message after import, got %d", len(ctx.Messages))
	}
	if um, ok := ctx.Messages[0].(ai.UserMessage); !ok || um.Content != "persist me" {
		t.Fatalf("imported message mismatch: %+v", ctx.Messages[0])
	}
}

func TestSetLabel(t *testing.T) {
	dir := t.TempDir()
	mgr := NewSessionManager(dir)
	sess, err := mgr.Create("labeltest")
	if err != nil {
		t.Fatal(err)
	}
	sess.AppendMessage(ai.UserMessage{Content: "x", Timestamp: ai.Now()})
	if err := sess.SetLabel("my label"); err != nil {
		t.Fatal(err)
	}
	// Reload and verify the label surfaces in List.
	infos, _ := mgr.List()
	for _, info := range infos {
		if info.ID == "labeltest" && info.Label == "my label" {
			return
		}
	}
	t.Fatal("label not persisted/found")
}
