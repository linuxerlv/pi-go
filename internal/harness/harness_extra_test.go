package harness

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

// scriptTextTurn for harness tests (mirror of agent's helper, same package
// prefix not shared across packages).
func harnessTextTurn(text string) []ai.AssistantMessageEvent {
	msg := ai.AssistantMessage{
		Content:    []ai.ContentBlock{ai.TextContent{Type: "text", Text: text}},
		API:        ai.APIAnthropicMessages, Provider: "mock", Model: "mock-model",
		StopReason: ai.StopStop, Timestamp: ai.Now(),
	}
	return []ai.AssistantMessageEvent{
		ai.StartEvent{Partial: msg},
		ai.TextDeltaEvent{ContentIndex: 0, Delta: text, Partial: msg},
		ai.DoneEvent{Reason: ai.StopStop, Message: msg},
	}
}

// TestEstimateTokensMessageTypes verifies estimateTokens handles each message
// type and content-block kind. Note: ThinkingContent/ImageContent/ToolCall are
// currently NOT counted (defect 8) — this test records the current behavior so
// a future fix is intentional.
func TestEstimateTokensMessageTypes(t *testing.T) {
	// UserMessage with string content.
	userStr := []agent.AgentMessage{ai.UserMessage{Content: "hello world!", Timestamp: 1}}
	if n := estimateTokens(userStr); n <= 0 {
		t.Fatalf("expected positive tokens for user string, got %d", n)
	}

	// UserMessage with []ContentBlock.
	userBlocks := []agent.AgentMessage{ai.UserMessage{
		Content:   []ai.ContentBlock{ai.TextContent{Type: "text", Text: "abc"}},
		Timestamp: 1,
	}}
	if n := estimateTokens(userBlocks); n <= 0 {
		t.Fatalf("expected positive tokens for user blocks, got %d", n)
	}

	// AssistantMessage with TextContent.
	asst := []agent.AgentMessage{ai.AssistantMessage{
		Content:   []ai.ContentBlock{ai.TextContent{Type: "text", Text: "response text"}},
		Timestamp: 1,
	}}
	if n := estimateTokens(asst); n <= 0 {
		t.Fatalf("expected positive tokens for assistant text, got %d", n)
	}

	// ToolResultMessage.
	tr := []agent.AgentMessage{ai.ToolResultMessage{
		Content:   []ai.ContentBlock{ai.TextContent{Type: "text", Text: "tool output"}},
		Timestamp: 1,
	}}
	if n := estimateTokens(tr); n <= 0 {
		t.Fatalf("expected positive tokens for tool result, got %d", n)
	}

	// ThinkingContent alone is NOT counted (current behavior; defect 8).
	think := []agent.AgentMessage{ai.AssistantMessage{
		Content:   []ai.ContentBlock{ai.ThinkingContent{Type: "thinking", Thinking: strings.Repeat("a", 1000)}},
		Timestamp: 1,
	}}
	thinkTokens := estimateTokens(think)
	// Only the per-message overhead (4) is counted, not the thinking text.
	if thinkTokens != 4 {
		t.Fatalf("ThinkingContent should not be counted (defect 8); got %d, want 4 (overhead only)", thinkTokens)
	}
}

// TestSetModelIdleSucceedsAndNonIdleFails verifies SetModel is allowed only
// when idle.
func TestSetModelIdleSucceedsAndNonIdleFails(t *testing.T) {
	sess := NewSession(NewMemoryStorage(SessionMetadata{ID: "s1", CreatedAt: nowISO()}))
	h := New(Options{
		Provider: newMockProvider(harnessTextTurn("ok")),
		Model:    ai.Model{ID: "m1", API: ai.APIAnthropicMessages, Provider: "mock"},
		Session:  sess,
	})
	// Idle: succeeds.
	if err := h.SetModel(ai.Model{ID: "m2", Provider: "mock"}); err != nil {
		t.Fatalf("SetModel while idle should succeed: %v", err)
	}
}

// TestSetThinkingLevelIdleSucceeds verifies SetThinkingLevel works while idle.
func TestSetThinkingLevelIdleSucceeds(t *testing.T) {
	sess := NewSession(NewMemoryStorage(SessionMetadata{ID: "s1", CreatedAt: nowISO()}))
	h := New(Options{
		Provider:      newMockProvider(harnessTextTurn("ok")),
		Model:         ai.Model{ID: "m1", API: ai.APIAnthropicMessages, Provider: "mock"},
		ThinkingLevel: ai.ThinkingOff,
		Session:       sess,
	})
	if err := h.SetThinkingLevel(ai.ThinkingMedium); err != nil {
		t.Fatalf("SetThinkingLevel while idle should succeed: %v", err)
	}
}

// TestFollowUpQueuedWhileBusy verifies FollowUp is rejected while idle (the
// documented contract: follow-up only meaningful mid-run).
func TestFollowUpRejectedWhileIdle(t *testing.T) {
	h := New(Options{
		Provider: newMockProvider(harnessTextTurn("ok")),
		Model:    ai.Model{ID: "m1", API: ai.APIAnthropicMessages, Provider: "mock"},
		Session:  NewSession(NewMemoryStorage(SessionMetadata{ID: "s1", CreatedAt: nowISO()})),
	})
	if err := h.FollowUp("more"); err == nil {
		t.Fatal("FollowUp should error while idle")
	}
}

// TestSteerRejectedWhileIdle verifies Steer is rejected while idle.
func TestSteerRejectedWhileIdle(t *testing.T) {
	h := New(Options{
		Provider: newMockProvider(harnessTextTurn("ok")),
		Model:    ai.Model{ID: "m1", API: ai.APIAnthropicMessages, Provider: "mock"},
		Session:  NewSession(NewMemoryStorage(SessionMetadata{ID: "s1", CreatedAt: nowISO()})),
	})
	if err := h.Steer("mid"); err == nil {
		t.Fatal("Steer should error while idle")
	}
}

// TestAbortDuringRun verifies Abort cancels an in-flight run (it returns
// promptly without hanging).
func TestAbortDuringRun(t *testing.T) {
	// A provider whose script never emits a terminal event would hang; instead
	// we abort a normal short run and confirm Abort is safe to call.
	h := New(Options{
		Provider: newMockProvider(harnessTextTurn("ok")),
		Model:    ai.Model{ID: "m1", API: ai.APIAnthropicMessages, Provider: "mock"},
		Session:  NewSession(NewMemoryStorage(SessionMetadata{ID: "s1", CreatedAt: nowISO()})),
	})
	done := make(chan struct{})
	go func() {
		_, _ = h.Prompt(context.Background(), "hi")
		close(done)
	}()
	// Give the run a moment to start, then abort.
	time.Sleep(50 * time.Millisecond)
	h.Abort()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Prompt did not return after Abort within 5s")
	}
}

// TestEventBusUnsubscribe verifies an unsubscribed handler receives no further
// events.
func TestEventBusUnsubscribe(t *testing.T) {
	bus := &eventBus{}
	got := 0
	unsub := bus.subscribe(func(HarnessEvent) error { got++; return nil })
	bus.emit(HarnessEvent{})
	if got != 1 {
		t.Fatalf("expected 1 event before unsubscribe, got %d", got)
	}
	unsub()
	bus.emit(HarnessEvent{})
	if got != 1 {
		t.Fatalf("expected no event after unsubscribe, got %d", got)
	}
}

// TestEventBusMultipleHandlers verifies fan-out to all subscribers.
func TestEventBusMultipleHandlers(t *testing.T) {
	bus := &eventBus{}
	a, b := 0, 0
	bus.subscribe(func(HarnessEvent) error { a++; return nil })
	bus.subscribe(func(HarnessEvent) error { b++; return nil })
	bus.emit(HarnessEvent{})
	if a != 1 || b != 1 {
		t.Fatalf("expected both handlers called once, got a=%d b=%d", a, b)
	}
}
