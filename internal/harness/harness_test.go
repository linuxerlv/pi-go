package harness

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

// scriptTextTurn builds a script for a single assistant turn that emits text
// and finishes with stop.
func scriptTextTurn(text string) []ai.AssistantMessageEvent {
	msg := ai.AssistantMessage{
		Content:    []ai.ContentBlock{ai.TextContent{Type: "text", Text: text}},
		API:        ai.APIAnthropicMessages,
		Provider:   "mock",
		Model:      "mock-model",
		StopReason: ai.StopStop,
		Timestamp:  ai.Now(),
	}
	return []ai.AssistantMessageEvent{
		ai.StartEvent{Partial: msg},
		ai.TextStartEvent{ContentIndex: 0, Partial: msg},
		ai.TextDeltaEvent{ContentIndex: 0, Delta: text, Partial: msg},
		ai.TextEndEvent{ContentIndex: 0, Content: text, Partial: msg},
		ai.DoneEvent{Reason: ai.StopStop, Message: msg},
	}
}

func TestHarnessPromptPersistsMessages(t *testing.T) {
	prov := newMockProvider(scriptTextTurn("Hello!"))
	sess := NewSession(NewMemoryStorage(SessionMetadata{ID: "s1", CreatedAt: nowISO()}))

	h := New(Options{
		Provider:     prov,
		Model:        ai.Model{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "mock"},
		SystemPrompt: "you are a test bot",
		Session:      sess,
	})

	var events []agent.AgentEvent
	h.Subscribe(func(e HarnessEvent) error {
		if e.Agent != nil {
			events = append(events, e.Agent)
		}
		return nil
	})

	if _, err := h.Prompt(context.Background(), "hi"); err != nil {
		t.Fatalf("Prompt failed: %v", err)
	}

	// Session should contain the user prompt + the assistant message.
	ctx := sess.BuildContext()
	if len(ctx.Messages) != 2 {
		t.Fatalf("expected 2 messages in session, got %d", len(ctx.Messages))
	}
	if _, ok := ctx.Messages[0].(ai.UserMessage); !ok {
		t.Fatalf("expected first message to be user, got %T", ctx.Messages[0])
	}
	if am, ok := ctx.Messages[1].(ai.AssistantMessage); !ok || am.StopReason != ai.StopStop {
		t.Fatalf("expected second message to be a stopped assistant message, got %T %+v", ctx.Messages[1], ctx.Messages[1])
	}

	// Event sequence should include agent_start and agent_end.
	if len(events) == 0 {
		t.Fatal("no events emitted")
	}
	if events[0].EventType() != "agent_start" {
		t.Fatalf("expected first event agent_start, got %s", events[0].EventType())
	}
	if events[len(events)-1].EventType() != "agent_end" {
		t.Fatalf("expected last event agent_end, got %s", events[len(events)-1].EventType())
	}
}

func TestJsonlStorageRoundTrip(t *testing.T) {
	dir := t.TempDir()
	s1, err := NewJsonlStorage(dir, "session-1")
	if err != nil {
		t.Fatal(err)
	}
	sess1 := NewSession(s1)

	// Append a user + assistant message via a harness-like flow.
	userMsg := ai.UserMessage{Content: "hello", Timestamp: ai.Now()}
	_, err = sess1.AppendMessage(userMsg)
	if err != nil {
		t.Fatalf("AppendMessage user: %v", err)
	}
	asstMsg := ai.AssistantMessage{
		Content:    []ai.ContentBlock{ai.TextContent{Type: "text", Text: "hi back"}},
		API:        ai.APIAnthropicMessages,
		Provider:   "mock",
		Model:      "mock-model",
		StopReason: ai.StopStop,
		Timestamp:  ai.Now(),
	}
	_, err = sess1.AppendMessage(asstMsg)
	if err != nil {
		t.Fatalf("AppendMessage assistant: %v", err)
	}

	// Verify the jsonl file exists and has 2 lines.
	b, err := os.ReadFile(filepath.Join(dir, "session-1.jsonl"))
	if err != nil {
		t.Fatalf("read jsonl: %v", err)
	}
	if got := countLines(b); got != 2 {
		t.Fatalf("expected 2 jsonl lines, got %d", got)
	}

	// Reload from disk and verify the branch rebuilds the same messages.
	s2, err := NewJsonlStorage(dir, "session-1")
	if err != nil {
		t.Fatal(err)
	}
	sess2 := NewSession(s2)
	ctx := sess2.BuildContext()
	if len(ctx.Messages) != 2 {
		t.Fatalf("expected 2 messages after reload, got %d", len(ctx.Messages))
	}
	if um, ok := ctx.Messages[0].(ai.UserMessage); !ok {
		t.Fatalf("expected user message first, got %T", ctx.Messages[0])
	} else if um.Content != "hello" {
		t.Fatalf("expected user content 'hello', got %v", um.Content)
	}
	if am, ok := ctx.Messages[1].(ai.AssistantMessage); !ok {
		t.Fatalf("expected assistant message second, got %T", ctx.Messages[1])
	} else if len(am.Content) == 0 {
		t.Fatal("assistant message has no content")
	}
}

func TestCompactionRebuildDropsOldEntries(t *testing.T) {
	sess := NewSession(NewMemoryStorage(SessionMetadata{ID: "s1", CreatedAt: nowISO()}))
	// Three messages.
	for _, m := range []ai.Message{
		ai.UserMessage{Content: "old1", Timestamp: ai.Now()},
		ai.AssistantMessage{Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: "old2"}}, Provider: "mock", Model: "m", StopReason: ai.StopStop, Timestamp: ai.Now()},
		ai.UserMessage{Content: "recent", Timestamp: ai.Now()},
	} {
		if _, err := sess.AppendMessage(m); err != nil {
			t.Fatal(err)
		}
	}
	entries := sess.Storage().AllEntries()
	if len(entries) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(entries))
	}

	// Insert a compaction entry that keeps only the 3rd message, with a summary.
	keep := entries[2].Base().ID
	compaction := CompactionEntry{
		EntryBase: EntryBase{
			Type:      EntryCompaction,
			ID:        sess.Storage().CreateEntryID(),
			ParentID:  sess.GetLeafID(),
			Timestamp: nowISO(),
		},
		Summary:          "compacted old convo",
		FirstKeptEntryID: keep,
		TokensBefore:     100,
	}
	if err := sess.Storage().AppendEntry(compaction); err != nil {
		t.Fatal(err)
	}
	leaf := compaction.ID
	_ = sess.Storage().SetLeafID(&leaf)

	ctx := sess.BuildContext()
	// Expect: compaction summary + the kept message (recent). old1/old2 dropped.
	if len(ctx.Messages) != 2 {
		t.Fatalf("expected 2 messages after compaction (summary + kept), got %d", len(ctx.Messages))
	}
	// First should be the compaction summary user message.
	if um, ok := ctx.Messages[0].(ai.UserMessage); !ok {
		t.Fatalf("expected compaction summary first, got %T", ctx.Messages[0])
	} else {
		s, _ := um.Content.(string)
		if s == "" || s != "[compaction summary] compacted old convo" {
			t.Fatalf("unexpected summary: %v", um.Content)
		}
	}
}

func countLines(b []byte) int {
	n := 0
	for _, c := range b {
		if c == '\n' {
			n++
		}
	}
	return n
}
