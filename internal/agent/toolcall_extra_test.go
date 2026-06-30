package agent

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// recordingTool is a fakeTool that records its execution order and supports a
// sequential execution-mode override and an optional PrepareArguments error.
type recordingTool struct {
	name      string
	execMode  ToolExecutionMode
	prepErr   error
	mu        sync.Mutex
	execOrder []int
	call      int
}

var execCounter struct {
	sync.Mutex
	n int
}

func (t *recordingTool) Def() ai.Tool {
	return ai.Tool{
		Name:        t.name,
		Description: "rec " + t.name,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{"x": map[string]any{"type": "string"}},
			"required":   []any{"x"},
		},
	}
}
func (t *recordingTool) Label() string { return t.name }
func (t *recordingTool) PrepareArguments(args map[string]any) (map[string]any, error) {
	if t.prepErr != nil {
		return nil, t.prepErr
	}
	return args, nil
}
func (t *recordingTool) Execute(ctx context.Context, id string, params map[string]any, onUpdate AgentToolUpdateCallback) (AgentToolResult, error) {
	execCounter.Lock()
	execCounter.n++
	n := execCounter.n
	execCounter.Unlock()
	t.mu.Lock()
	t.execOrder = append(t.execOrder, n)
	t.mu.Unlock()
	return AgentToolResult{Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: t.name}}}, nil
}
func (t *recordingTool) ExecutionMode() ToolExecutionMode { return t.execMode }

// TestSequentialExecutionRunsInOrder verifies that with ToolExecutionSequential
// (or a tool declaring sequential), a batch of tool calls executes one at a time
// in assistant source order.
func TestSequentialExecutionRunsInOrder(t *testing.T) {
	execCounter.Lock()
	execCounter.n = 0
	execCounter.Unlock()
	a := &recordingTool{name: "a", execMode: ToolExecutionSequential}
	b := &recordingTool{name: "b", execMode: ToolExecutionSequential}

	// One turn with two sequential tool calls, then a final text turn.
	tcA := ai.ToolCall{Type: "toolCall", ID: "c1", Name: "a", Arguments: map[string]any{"x": "1"}, ArgumentsRaw: []byte(`{"x":"1"}`)}
	tcB := ai.ToolCall{Type: "toolCall", ID: "c2", Name: "b", Arguments: map[string]any{"x": "2"}, ArgumentsRaw: []byte(`{"x":"2"}`)}
	toolMsg := ai.AssistantMessage{
		Content:    []ai.ContentBlock{tcA, tcB},
		API:        ai.APIAnthropicMessages, Provider: "mock", Model: "mock-model",
		StopReason: ai.StopToolUse, Timestamp: ai.Now(),
	}
	toolTurn := []ai.AssistantMessageEvent{
		ai.StartEvent{Partial: toolMsg},
		ai.ToolCallEndEvent{ContentIndex: 0, ToolCall: tcA, Partial: toolMsg},
		ai.ToolCallEndEvent{ContentIndex: 1, ToolCall: tcB, Partial: toolMsg},
		ai.DoneEvent{Reason: ai.StopToolUse, Message: toolMsg},
	}
	prov := newMockProvider(toolTurn, scriptTextTurn("done"))

	config := NewLoopConfig(ai.Model{ID: "mock-model"}).
		WithConvertToLlm(DefaultConvertToLlm).
		WithToolExecution(ToolExecutionSequential).
		Build()
	ctx := AgentContext{SystemPrompt: "sys", Tools: []AgentTool{a, b}}
	_, err := RunAgentLoop(context.Background(),
		[]AgentMessage{ai.UserMessage{Content: "do", Timestamp: ai.Now()}},
		ctx, config, prov, func(AgentEvent) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	// a must have executed before b (lower global counter).
	a.mu.Lock()
	b.mu.Lock()
	if len(a.execOrder) != 1 || len(b.execOrder) != 1 {
		t.Fatalf("expected each tool executed once, got a=%v b=%v", a.execOrder, b.execOrder)
	}
	if a.execOrder[0] >= b.execOrder[0] {
		t.Fatalf("expected a before b, got a=%d b=%d", a.execOrder[0], b.execOrder[0])
	}
	b.mu.Unlock()
	a.mu.Unlock()
}

// TestShouldTerminateBatch verifies the all-true semantics.
func TestShouldTerminateBatch(t *testing.T) {
	if shouldTerminateBatch(nil) {
		t.Fatal("empty batch should not terminate")
	}
	all := []finalizedToolCall{
		{result: AgentToolResult{Terminate: true}},
		{result: AgentToolResult{Terminate: true}},
	}
	if !shouldTerminateBatch(all) {
		t.Fatal("all-terminate batch should terminate")
	}
	mixed := []finalizedToolCall{
		{result: AgentToolResult{Terminate: true}},
		{result: AgentToolResult{Terminate: false}},
	}
	if shouldTerminateBatch(mixed) {
		t.Fatal("mixed batch should not terminate")
	}
}

// TestPrepareArgumentsErrorYieldsImmediateErrorResult verifies that a
// Preparer returning an error short-circuits execution with an error result.
func TestPrepareArgumentsErrorYieldsImmediateErrorResult(t *testing.T) {
	tool := &recordingTool{name: "bad", prepErr: errors.New("prep failed")}
	prov := newMockProvider(
		scriptToolCallTurn("c1", "bad", map[string]any{"x": "1"}),
		scriptTextTurn("done"),
	)
	var sawError bool
	emit := func(e AgentEvent) error {
		if te, ok := e.(ToolExecutionEndEvent); ok && te.ToolName == "bad" && te.IsError {
			sawError = true
		}
		return nil
	}
	ctx := AgentContext{SystemPrompt: "sys", Tools: []AgentTool{tool}}
	_, err := RunAgentLoop(context.Background(),
		[]AgentMessage{ai.UserMessage{Content: "x", Timestamp: ai.Now()}},
		ctx, NewLoopConfig(ai.Model{ID: "m"}).WithConvertToLlm(DefaultConvertToLlm).Build(),
		prov, emit)
	if err != nil {
		t.Fatal(err)
	}
	if !sawError {
		t.Fatal("expected error tool result from PrepareArguments failure")
	}
	// The tool's Execute must never run.
	tool.mu.Lock()
	ran := len(tool.execOrder)
	tool.mu.Unlock()
	if ran != 0 {
		t.Fatalf("Execute must not run on prep error, ran %d", ran)
	}
}

// TestValidationFailureYieldsImmediateErrorResult verifies that invalid tool
// args (schema violation) short-circuit execution.
func TestValidationFailureYieldsImmediateErrorResult(t *testing.T) {
	tool := &recordingTool{name: "strict"} // requires "x" string
	prov := newMockProvider(
		// args missing required "x".
		scriptToolCallTurn("c1", "strict", map[string]any{}),
		scriptTextTurn("done"),
	)
	var sawError bool
	emit := func(e AgentEvent) error {
		if te, ok := e.(ToolExecutionEndEvent); ok && te.ToolName == "strict" && te.IsError {
			sawError = true
		}
		return nil
	}
	ctx := AgentContext{SystemPrompt: "sys", Tools: []AgentTool{tool}}
	_, err := RunAgentLoop(context.Background(),
		[]AgentMessage{ai.UserMessage{Content: "x", Timestamp: ai.Now()}},
		ctx, NewLoopConfig(ai.Model{ID: "m"}).WithConvertToLlm(DefaultConvertToLlm).Build(),
		prov, emit)
	if err != nil {
		t.Fatal(err)
	}
	if !sawError {
		t.Fatal("expected error tool result from validation failure")
	}
}

// TestPartialFromEvent verifies the event->partial extraction (pure function).
func TestPartialFromEvent(t *testing.T) {
	fallback := ai.AssistantMessage{Provider: "fallback"}
	textMsg := ai.AssistantMessage{Provider: "text", Model: "m"}
	cases := []ai.AssistantMessageEvent{
		ai.TextStartEvent{Partial: textMsg},
		ai.TextDeltaEvent{Partial: textMsg},
		ai.TextEndEvent{Partial: textMsg},
		ai.ThinkingStartEvent{Partial: textMsg},
		ai.ThinkingDeltaEvent{Partial: textMsg},
		ai.ThinkingEndEvent{Partial: textMsg},
		ai.ToolCallStartEvent{Partial: textMsg},
		ai.ToolCallDeltaEvent{Partial: textMsg},
		ai.ToolCallEndEvent{Partial: textMsg},
	}
	for i, e := range cases {
		got := partialFromEvent(e, fallback)
		if got.Provider != "text" {
			t.Fatalf("case %d: expected partial from event, got provider %q", i, got.Provider)
		}
	}
	// Unknown event type returns fallback.
	unknown := ai.StartEvent{Partial: ai.AssistantMessage{Provider: "start"}}
	if got := partialFromEvent(unknown, fallback); got.Provider != "fallback" {
		t.Fatalf("expected fallback for StartEvent, got %q", got.Provider)
	}
}

// TestRunAgentLoopCancellable verifies the loop honors context cancellation.
func TestRunAgentLoopCancellable(t *testing.T) {
	// A provider script that blocks forever on the second turn would hang; instead
	// cancel before starting and confirm the loop returns promptly.
	prov := newMockProvider(scriptTextTurn("hi"))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	// With a cancelled ctx, the loop should return ctx.Err() (or complete the
	// first turn before noticing). Use a timeout to bound.
	done := make(chan error, 1)
	go func() {
		_, err := RunAgentLoop(ctx,
			[]AgentMessage{ai.UserMessage{Content: "x", Timestamp: ai.Now()}},
			AgentContext{SystemPrompt: "sys"},
			NewLoopConfig(ai.Model{ID: "m"}).WithConvertToLlm(DefaultConvertToLlm).Build(),
			prov, func(AgentEvent) error { return nil })
		done <- err
	}()
	select {
	case <-done:
		// ok: returned (either error or nil) without hanging
	case <-time.After(5 * time.Second):
		t.Fatal("loop did not return after ctx cancel within 5s")
	}
}
