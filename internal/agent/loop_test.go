package agent

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// fakeTool is a controllable AgentTool for tests.
type fakeTool struct {
	name   string
	params map[string]any
	result AgentToolResult
	err    error
}

func (t *fakeTool) Def() ai.Tool {
	return ai.Tool{
		Name:        t.name,
		Description: "fake " + t.name,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"x": map[string]any{"type": "string"},
			},
		},
	}
}
func (t *fakeTool) Label() string                   { return t.name }
func (t *fakeTool) PrepareArguments(args map[string]any) (map[string]any, error) {
	return args, nil
}
func (t *fakeTool) Execute(ctx context.Context, id string, params map[string]any, onUpdate AgentToolUpdateCallback) (AgentToolResult, error) {
	return t.result, t.err
}
func (t *fakeTool) ExecutionMode() ToolExecutionMode { return "" }

func TestRunAgentLoopSingleTextTurn(t *testing.T) {
	prov := newMockProvider(scriptTextTurn("hello"))
	var events []AgentEvent
	emit := func(e AgentEvent) error { events = append(events, e); return nil }

	final, err := RunAgentLoop(context.Background(),
		[]AgentMessage{ai.UserMessage{Content: "hi", Timestamp: ai.Now()}},
		AgentContext{SystemPrompt: "sys"},
		AgentLoopConfig{Model: ai.Model{ID: "mock-model"}, ConvertToLlm: DefaultConvertToLlm},
		prov, emit)
	if err != nil {
		t.Fatal(err)
	}
	if len(final) != 2 {
		t.Fatalf("expected 2 new messages (prompt + assistant), got %d", len(final))
	}
	// Event sequence: agent_start, turn_start, message_start(user), message_end(user),
	// message_start(assistant), message_update, message_end, turn_end, agent_end.
	if events[0].EventType() != "agent_start" {
		t.Fatalf("expected agent_start first, got %s", events[0].EventType())
	}
	if events[len(events)-1].EventType() != "agent_end" {
		t.Fatalf("expected agent_end last, got %s", events[len(events)-1].EventType())
	}
}

func TestRunAgentLoopToolCallRoundTrip(t *testing.T) {
	tool := &fakeTool{name: "read", result: AgentToolResult{
		Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: "file contents"}},
	}}
	prov := newMockProvider(
		scriptToolCallTurn("call_1", "read", map[string]any{"x": "f"}),
		scriptTextTurn("done"),
	)
	var events []AgentEvent
	emit := func(e AgentEvent) error { events = append(events, e); return nil }

	ctx := AgentContext{
		SystemPrompt: "sys",
		Tools:        []AgentTool{tool},
	}
	_, err := RunAgentLoop(context.Background(),
		[]AgentMessage{ai.UserMessage{Content: "read f", Timestamp: ai.Now()}},
		ctx,
		AgentLoopConfig{Model: ai.Model{ID: "mock-model"}, ConvertToLlm: DefaultConvertToLlm},
		prov, emit)
	if err != nil {
		t.Fatal(err)
	}
	// Must have seen a tool_execution_end for "read".
	var sawToolEnd bool
	for _, e := range events {
		if te, ok := e.(ToolExecutionEndEvent); ok && te.ToolName == "read" {
			sawToolEnd = true
			if te.IsError {
				t.Fatalf("tool should not error")
			}
		}
	}
	if !sawToolEnd {
		t.Fatal("expected tool_execution_end for read")
	}
	// Context must retain the assistant tool_use message (the bug fixed in
	// Task 1): the second turn's LLM call sees user + assistant(tool_use) +
	// toolResult. Verify by checking the provider received a context with 3
	// messages on the second call. We approximate by confirming the loop ran
	// both scripts.
	if len(prov.scripts) != 0 {
		t.Fatal("expected both scripts consumed")
	}
}

func TestRunAgentLoopErrorTurnExits(t *testing.T) {
	prov := newMockProvider(scriptErrorTurn("boom"))
	var events []AgentEvent
	emit := func(e AgentEvent) error { events = append(events, e); return nil }

	_, err := RunAgentLoop(context.Background(),
		[]AgentMessage{ai.UserMessage{Content: "hi", Timestamp: ai.Now()}},
		AgentContext{SystemPrompt: "sys"},
		AgentLoopConfig{Model: ai.Model{ID: "mock-model"}, ConvertToLlm: DefaultConvertToLlm},
		prov, emit)
	if err != nil {
		t.Fatalf("error turns are encoded in-stream, not returned: %v", err)
	}
	// agent_end must follow turn_end without a second turn.
	var last string
	for _, e := range events {
		last = e.EventType()
	}
	if last != "agent_end" {
		t.Fatalf("expected agent_end last, got %s", last)
	}
}

func TestRunAgentLoopToolNotFound(t *testing.T) {
	// Assistant calls a tool that doesn't exist; loop should emit an error
	// tool result and continue. Provide a second turn so the loop completes.
	prov := newMockProvider(
		scriptToolCallTurn("c1", "ghost", map[string]any{}),
		scriptTextTurn("ok"),
	)
	var events []AgentEvent
	emit := func(e AgentEvent) error { events = append(events, e); return nil }

	ctx := AgentContext{SystemPrompt: "sys", Tools: []AgentTool{}} // no tools
	_, err := RunAgentLoop(context.Background(),
		[]AgentMessage{ai.UserMessage{Content: "x", Timestamp: ai.Now()}},
		ctx,
		AgentLoopConfig{Model: ai.Model{ID: "mock-model"}, ConvertToLlm: DefaultConvertToLlm},
		prov, emit)
	if err != nil {
		t.Fatal(err)
	}
	var sawErrorResult bool
	for _, e := range events {
		if te, ok := e.(ToolExecutionEndEvent); ok && te.ToolName == "ghost" && te.IsError {
			sawErrorResult = true
		}
	}
	if !sawErrorResult {
		t.Fatal("expected error tool result for missing tool")
	}
}

func TestRunAgentLoopBeforeToolCallBlock(t *testing.T) {
	tool := &fakeTool{name: "read", result: AgentToolResult{
		Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: "ok"}},
	}}
	prov := newMockProvider(
		scriptToolCallTurn("c1", "read", map[string]any{"x": "f"}),
		scriptTextTurn("done"),
	)
	blocked := false
	config := AgentLoopConfig{
		Model:        ai.Model{ID: "mock-model"},
		ConvertToLlm: DefaultConvertToLlm,
		BeforeToolCall: func(ctx context.Context, c BeforeToolCallContext) (*BeforeToolCallResult, error) {
			blocked = true
			return &BeforeToolCallResult{Block: true, Reason: "nope"}, nil
		},
	}
	ctx := AgentContext{SystemPrompt: "sys", Tools: []AgentTool{tool}}
	_, err := RunAgentLoop(context.Background(),
		[]AgentMessage{ai.UserMessage{Content: "x", Timestamp: ai.Now()}},
		ctx, config, prov,
		func(AgentEvent) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if !blocked {
		t.Fatal("BeforeToolCall should have been called")
	}
}

// TestParallelAfterToolCallNotSerialized verifies that the AfterToolCall hook
// runs concurrently across tools in a parallel batch. Previously the hook was
// invoked under the batch mutex, serializing all tools. We measure max in-flight
// concurrency of the hook; with the fix it reaches 2.
func TestParallelAfterToolCallNotSerialized(t *testing.T) {
	toolA := &fakeTool{name: "a", result: AgentToolResult{
		Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: "a"}},
	}}
	toolB := &fakeTool{name: "b", result: AgentToolResult{
		Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: "b"}},
	}}

	// Script one turn emitting two tool calls, then a final text turn.
	tcA := ai.ToolCall{Type: "toolCall", ID: "c1", Name: "a", Arguments: map[string]any{"x": "1"}, ArgumentsRaw: []byte(`{"x":"1"}`)}
	tcB := ai.ToolCall{Type: "toolCall", ID: "c2", Name: "b", Arguments: map[string]any{"x": "1"}, ArgumentsRaw: []byte(`{"x":"1"}`)}
	toolMsg := ai.AssistantMessage{
		Content:    []ai.ContentBlock{tcA, tcB},
		API:        ai.APIAnthropicMessages, Provider: "mock", Model: "mock-model",
		StopReason: ai.StopToolUse, Timestamp: ai.Now(),
	}
	toolTurn := []ai.AssistantMessageEvent{
		ai.StartEvent{Partial: toolMsg},
		ai.ToolCallStartEvent{ContentIndex: 0, Partial: toolMsg},
		ai.ToolCallEndEvent{ContentIndex: 0, ToolCall: tcA, Partial: toolMsg},
		ai.ToolCallStartEvent{ContentIndex: 1, Partial: toolMsg},
		ai.ToolCallEndEvent{ContentIndex: 1, ToolCall: tcB, Partial: toolMsg},
		ai.DoneEvent{Reason: ai.StopToolUse, Message: toolMsg},
	}
	prov := newMockProvider(toolTurn, scriptTextTurn("done"))

	var inFlight, maxInFlight int32
	afterFn := func(ctx context.Context, c AfterToolCallContext) (*AfterToolCallResult, error) {
		n := atomic.AddInt32(&inFlight, 1)
		for {
			m := atomic.LoadInt32(&maxInFlight)
			if n <= m {
				break
			}
			if atomic.CompareAndSwapInt32(&maxInFlight, m, n) {
				break
			}
		}
		time.Sleep(50 * time.Millisecond) // widen the overlap window
		atomic.AddInt32(&inFlight, -1)
		return nil, nil
	}

	config := NewLoopConfig(ai.Model{ID: "mock-model"}).
		WithConvertToLlm(DefaultConvertToLlm).
		WithPermission(nil, afterFn).
		Build()

	ctx := AgentContext{SystemPrompt: "sys", Tools: []AgentTool{toolA, toolB}}
	_, err := RunAgentLoop(context.Background(),
		[]AgentMessage{ai.UserMessage{Content: "do both", Timestamp: ai.Now()}},
		ctx, config, prov,
		func(AgentEvent) error { return nil })
	if err != nil {
		t.Fatal(err)
	}

	// Both AfterToolCall hooks must have overlapped. If they ran serialized
	// (the bug), maxInFlight stays at 1.
	if got := atomic.LoadInt32(&maxInFlight); got < 2 {
		t.Fatalf("AfterToolCall should run concurrently; max in-flight=%d want >=2", got)
	}
}
