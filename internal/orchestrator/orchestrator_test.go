package orchestrator

import (
	"context"
	"strings"
	"testing"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// mockProviderStream is a simple provider that returns scripted text.
type mockStreamProvider struct {
	ai.Provider
}

func (p *mockStreamProvider) ID() string      { return "mock-orch" }
func (p *mockStreamProvider) Name() string    { return "MockOrch" }
func (p *mockStreamProvider) BaseURL() string { return "" }
func (p *mockStreamProvider) Models() []ai.Model {
	return []ai.Model{{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "mock-orch"}}
}
func (p *mockStreamProvider) GetModel(id string) (ai.Model, bool) {
	return ai.Model{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "mock-orch"}, true
}
func (p *mockStreamProvider) StreamSimple(ctx context.Context, model ai.Model, context ai.Context, options *ai.SimpleStreamOptions) ai.AssistantMessageEventStream {
	stream := ai.NewAssistantMessageEventStream()
	go func() {
		defer stream.End(nil)
		// Collect user prompt text from context for subtask parsing.
		userText := ""
		for _, m := range context.Messages {
			if um, ok := m.(ai.UserMessage); ok {
				if s, ok := um.Content.(string); ok {
					userText = s
				}
			}
		}
		// Since the mock can't do LLM planning, feed a canned JSON subtask list.
		if userText == "" {
			return
		}
		// The first call is always planning; return a single subtask.
		msg := ai.AssistantMessage{
			Content:    []ai.ContentBlock{ai.TextContent{Type: "text", Text: `[{"id":"1","description":"solve the task"}]`}},
			API:        ai.APIAnthropicMessages,
			Provider:   "mock-orch",
			Model:      "mock-model",
			StopReason: ai.StopStop,
			Timestamp:  ai.Now(),
		}
		stream.Push(ai.StartEvent{Partial: msg})
		stream.Push(ai.TextStartEvent{ContentIndex: 0, Partial: msg})
		stream.Push(ai.TextDeltaEvent{ContentIndex: 0, Delta: `[{"id":"1","description":"solve the task"}]`, Partial: msg})
		stream.Push(ai.TextEndEvent{ContentIndex: 0, Content: `[{"id":"1","description":"solve the task"}]`, Partial: msg})
		stream.Push(ai.DoneEvent{Reason: ai.StopStop, Message: msg})
	}()
	return stream
}

func TestOrchestratorSequential(t *testing.T) {
	prov := &mockStreamProvider{}
	orch := New(Options{
		Provider: prov,
		Model:    ai.Model{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "mock-orch"},
		Strategy: &SequentialStrategy{},
	})
	result, err := orch.Run(context.Background(), "test task")
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if len(result.SubTasks) == 0 {
		t.Fatal("expected at least one subtask")
	}
	if result.FinalAnswer == "" {
		t.Fatal("expected non-empty final answer")
	}
	if result.Task != "test task" {
		t.Fatalf("expected task to be 'test task', got %q", result.Task)
	}
}

func TestParseSubTasks(t *testing.T) {
	tests := []struct {
		input string
		n     int
	}{
		{`[{"id":"1","description":"task one"}]`, 1},
		{`[{"id":"a","description":"x"},{"id":"b","description":"y"}]`, 2},
		{`plain text fallback`, 1},
	}
	for _, tc := range tests {
		tasks, err := parseSubTasks(tc.input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(tasks) != tc.n {
			t.Fatalf("expected %d tasks, got %d", tc.n, len(tasks))
		}
	}
}

func TestOrchestratorParallel(t *testing.T) {
	prov := &mockStreamProvider{}
	orch := New(Options{
		Provider: prov,
		Model:    ai.Model{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "mock-orch"},
		Strategy: &ParallelStrategy{MaxConcurrency: 2},
	})
	result, err := orch.Run(context.Background(), "parallel test")
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if len(result.SubTasks) == 0 {
		t.Fatal("expected at least one subtask")
	}
	if result.FinalAnswer == "" {
		t.Fatal("expected non-empty final answer")
	}
}

// TestParallelStrategyFailureRecordsPlaceholder verifies that when a subtask
// fails, its result slot is filled with a failure placeholder instead of being
// left as a zero-value empty SubTaskResult that synthesis would silently drop.
func TestParallelStrategyFailureRecordsPlaceholder(t *testing.T) {
	prov := &mockStreamProvider{}
	orch := New(Options{
		Provider: prov,
		Model:    ai.Model{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "mock-orch"},
		Strategy: &ParallelStrategy{MaxConcurrency: 2},
	})
	subtasks := []SubTask{
		{ID: "1", Description: "a"},
		{ID: "2", Description: "b"},
	}
	// Pre-cancel so RunSubtask fails fast (runLoop returns ctx.Err() at the top).
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	strat := &ParallelStrategy{MaxConcurrency: 2}
	results, err := strat.Execute(ctx, subtasks, orch)
	// Execute returns the first failure as err but still populates results.
	if err == nil {
		t.Fatal("expected Execute to return an error when subtasks fail")
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 result slots, got %d", len(results))
	}
	for i, r := range results {
		if r.SubTask.ID != subtasks[i].ID {
			t.Errorf("result[%d] subtask id = %q, want %q", i, r.SubTask.ID, subtasks[i].ID)
		}
		if !strings.Contains(r.Output, "failed") {
			t.Errorf("result[%d] output %q should contain a failure placeholder", i, r.Output)
		}
	}
}
