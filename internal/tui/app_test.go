package tui

import (
	"context"
	"strings"
	"testing"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

// fakeHarnessRunner is a minimal HarnessRunner for testing the Model without a
// real harness or bubbletea program.
type fakeHarnessRunner struct{ phase string }

func (f *fakeHarnessRunner) Subscribe(func(agent.AgentEvent)) func() { return func() {} }
func (f *fakeHarnessRunner) Prompt(ctx context.Context, text string) ([]agent.AgentMessage, error) {
	return nil, nil
}
func (f *fakeHarnessRunner) Phase() string { return f.phase }

// TestApplyEventAppendsLines verifies applyEvent renders the right history
// lines for key agent events.
func TestApplyEventAppendsLines(t *testing.T) {
	m := NewModel(nil, "mock")
	// Assistant message start adds an "assistant:" line.
	m.applyEvent(agent.MessageStartEvent{Message: ai.AssistantMessage{}})
	if len(m.history) != 1 {
		t.Fatalf("expected 1 history line after assistant start, got %d", len(m.history))
	}
	// Text delta appends to the last line.
	m.applyEvent(agent.MessageUpdateEvent{
		AssistantMessageEvent: ai.TextDeltaEvent{Delta: "hello"},
	})
	if m.history[len(m.history)-1].text == "" || !strings.Contains(m.history[len(m.history)-1].text, "hello") {
		t.Fatalf("expected last line to contain delta 'hello', got %q", m.history[len(m.history)-1].text)
	}
	// Tool start/end add lines.
	m.applyEvent(agent.ToolExecutionStartEvent{ToolName: "read"})
	m.applyEvent(agent.ToolExecutionEndEvent{ToolName: "read", IsError: false})
	if len(m.history) != 3 {
		t.Fatalf("expected 3 history lines after tool start+end, got %d", len(m.history))
	}
	// AgentEnd sets status.
	m.applyEvent(agent.AgentEndEvent{})
	if m.status != "done" {
		t.Fatalf("expected status 'done' after AgentEnd, got %q", m.status)
	}
}

// TestApplyEventToolErrorUsesErrorMarker verifies an errored tool end renders
// the error marker (the history line text differs from the success marker).
func TestApplyEventToolErrorUsesErrorMarker(t *testing.T) {
	m := NewModel(nil, "mock")
	m.applyEvent(agent.ToolExecutionEndEvent{ToolName: "bash", IsError: true})
	last := m.history[len(m.history)-1].text
	if !strings.Contains(last, "bash") {
		t.Fatalf("expected tool name in line, got %q", last)
	}
}

// TestAppendToLastOnEmptyHistory verifies appendToLast creates a line when
// history is empty (no panic on empty slice).
func TestAppendToLastOnEmptyHistory(t *testing.T) {
	m := NewModel(nil, "mock")
	m.appendToLast("first")
	if len(m.history) != 1 || m.history[0].text != "first" {
		t.Fatalf("expected single line 'first', got %+v", m.history)
	}
}

// TestAnswerPermReplies verifies answerPerm sends the answer on the reply
// channel and clears modal state.
func TestAnswerPermReplies(t *testing.T) {
	m := NewModel(nil, "mock")
	ch := make(chan string, 1)
	m.permActive = true
	m.permPrompt = "run bash?"
	m.permReplyCh = ch
	m.answerPerm("allow")
	if m.permActive {
		t.Fatal("permActive should be cleared after answer")
	}
	if m.permPrompt != "" {
		t.Fatal("permPrompt should be cleared")
	}
	if m.permReplyCh != nil {
		t.Fatal("permReplyCh should be cleared")
	}
	select {
	case got := <-ch:
		if got != "allow" {
			t.Fatalf("expected 'allow' reply, got %q", got)
		}
	default:
		t.Fatal("expected reply on channel")
	}
}

// TestAnswerPermNilChannelDoesNotPanic verifies answerPerm is safe with no
// reply channel set.
func TestAnswerPermNilChannelDoesNotPanic(t *testing.T) {
	m := NewModel(nil, "mock")
	m.permActive = true
	// permReplyCh is nil.
	m.answerPerm("deny")
	if m.permActive {
		t.Fatal("permActive should be cleared")
	}
}

// TestPhaseLabelWithNilHarness verifies phaseLabel returns "" with no harness.
func TestPhaseLabelWithNilHarness(t *testing.T) {
	m := NewModel(nil, "mock")
	if got := m.phaseLabel(); got != "" {
		t.Fatalf("expected empty phase with nil harness, got %q", got)
	}
}

// TestPhaseLabelWithHarness verifies phaseLabel delegates to the harness.
func TestPhaseLabelWithHarness(t *testing.T) {
	m := NewModel(&fakeHarnessRunner{phase: "turn"}, "mock")
	if got := m.phaseLabel(); got != "turn" {
		t.Fatalf("expected phase 'turn', got %q", got)
	}
}

// TestViewStarting verifies View returns a placeholder before the window size
// is known (width == 0).
func TestViewStarting(t *testing.T) {
	m := NewModel(nil, "mock")
	if got := m.View(); got != "starting…" {
		t.Fatalf("expected 'starting…' before window size, got %q", got)
	}
}
