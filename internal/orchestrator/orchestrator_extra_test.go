package orchestrator

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
	"github.com/linuxerlv/pi-go/internal/permission"
)

// scriptedProvider returns a different AssistantMessage script per StreamSimple
// call, in order. Unlike mockStreamProvider (which returns the same canned JSON
// for every call), this lets a test distinguish the plan/execute/synthesize
// calls and control each subagent's turns.
type scriptedProvider struct {
	mu       sync.Mutex
	scripts  [][]ai.AssistantMessageEvent // per-call scripts, consumed in order
	callIdx  int
}

func (p *scriptedProvider) ID() string      { return "scripted-orch" }
func (p *scriptedProvider) Name() string    { return "ScriptedOrch" }
func (p *scriptedProvider) BaseURL() string { return "" }
func (p *scriptedProvider) Models() []ai.Model {
	return []ai.Model{{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "scripted-orch"}}
}
func (p *scriptedProvider) GetModel(id string) (ai.Model, bool) {
	return ai.Model{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "scripted-orch"}, true
}
func (p *scriptedProvider) StreamSimple(ctx context.Context, model ai.Model, context ai.Context, options *ai.SimpleStreamOptions) ai.AssistantMessageEventStream {
	stream := ai.NewAssistantMessageEventStream()
	p.mu.Lock()
	idx := p.callIdx
	p.callIdx++
	scripts := p.scripts
	p.mu.Unlock()

	var script []ai.AssistantMessageEvent
	if idx < len(scripts) {
		script = scripts[idx]
	}
	go func() {
		defer stream.End(nil)
		for _, e := range script {
			stream.Push(e)
		}
	}()
	return stream
}

// textDoneScript builds a single-turn script that emits a text assistant
// message and finishes with stop.
func textDoneScript(text string) []ai.AssistantMessageEvent {
	msg := ai.AssistantMessage{
		Content:    []ai.ContentBlock{ai.TextContent{Type: "text", Text: text}},
		API:        ai.APIAnthropicMessages, Provider: "scripted-orch", Model: "mock-model",
		StopReason: ai.StopStop, Timestamp: ai.Now(),
	}
	return []ai.AssistantMessageEvent{
		ai.StartEvent{Partial: msg},
		ai.TextDeltaEvent{ContentIndex: 0, Delta: text, Partial: msg},
		ai.DoneEvent{Reason: ai.StopStop, Message: msg},
	}
}

// toolCallOnlyScript builds a script whose assistant message contains only a
// ToolCall (no TextContent) but stops with StopStop — so the loop does NOT
// execute the tool call, and RunSubtask's assistantText() yields "". This is
// the "produced no output" case (defect 3).
func toolCallOnlyScript(name string) []ai.AssistantMessageEvent {
	tc := ai.ToolCall{Type: "toolCall", ID: "c1", Name: name, Arguments: map[string]any{"x": "1"}, ArgumentsRaw: []byte(`{"x":"1"}`)}
	msg := ai.AssistantMessage{
		Content:    []ai.ContentBlock{tc},
		API:        ai.APIAnthropicMessages, Provider: "scripted-orch", Model: "mock-model",
		StopReason: ai.StopStop, Timestamp: ai.Now(),
	}
	return []ai.AssistantMessageEvent{
		ai.StartEvent{Partial: msg},
		ai.DoneEvent{Reason: ai.StopStop, Message: msg},
	}
}

// toolCallThenTextScript: turn 0 emits a tool call with StopToolUse (loop will
// execute it), turn 1 emits final text with StopStop. Used to exercise the
// permission wiring: the tool call is subject to BeforeToolCall.
func toolCallThenTextScript(toolName, finalText string) []ai.AssistantMessageEvent {
	tc := ai.ToolCall{Type: "toolCall", ID: "c1", Name: toolName, Arguments: map[string]any{"path": "f"}, ArgumentsRaw: []byte(`{"path":"f"}`)}
	toolMsg := ai.AssistantMessage{
		Content:    []ai.ContentBlock{tc},
		API:        ai.APIAnthropicMessages, Provider: "scripted-orch", Model: "mock-model",
		StopReason: ai.StopToolUse, Timestamp: ai.Now(),
	}
	return []ai.AssistantMessageEvent{
		ai.StartEvent{Partial: toolMsg},
		ai.ToolCallStartEvent{ContentIndex: 0, Partial: toolMsg},
		ai.ToolCallEndEvent{ContentIndex: 0, ToolCall: tc, Partial: toolMsg},
		ai.DoneEvent{Reason: ai.StopToolUse, Message: toolMsg},
	}
}

// fakeOrchTool is a controllable AgentTool for orchestrator tests.
type fakeOrchTool struct {
	name string
}

func (t *fakeOrchTool) Def() ai.Tool {
	return ai.Tool{
		Name:        t.name,
		Description: "fake " + t.name,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{"path": map[string]any{"type": "string"}},
			"required":   []any{"path"},
		},
	}
}
func (t *fakeOrchTool) Label() string { return t.name }
func (t *fakeOrchTool) Execute(ctx context.Context, id string, params map[string]any, onUpdate agent.AgentToolUpdateCallback) (agent.AgentToolResult, error) {
	return agent.AgentToolResult{
		Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: "ok"}},
	}, nil
}
func (t *fakeOrchTool) ExecutionMode() agent.ToolExecutionMode { return "" }

// TestRunSubtaskEmptyOutputErrors verifies that a subagent whose final
// assistant message has no text (only a tool_call, stopping with StopStop)
// surfaces "produced no output" instead of feeding an empty string to
// synthesis (defect 3).
func TestRunSubtaskEmptyOutputErrors(t *testing.T) {
	prov := &scriptedProvider{scripts: [][]ai.AssistantMessageEvent{
		toolCallOnlyScript("read"), // subagent turn: no text
	}}
	orch := New(Options{
		Provider: prov,
		Model:    ai.Model{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "scripted-orch"},
	})
	_, err := orch.RunSubtask(context.Background(), SubTask{ID: "1", Description: "do"})
	if err == nil {
		t.Fatal("expected error for empty subtask output")
	}
	if !strings.Contains(err.Error(), "produced no output") {
		t.Fatalf("expected 'produced no output' error, got: %v", err)
	}
}

// TestRunSubtaskPermissionWiring verifies that when Options.Permission is set,
// the subagent's tool calls go through the permission checker (defect 4). With
// ModePlan, a "write" tool call is blocked; the subagent still completes (the
// blocked tool yields an error result, then the second turn emits final text).
func TestRunSubtaskPermissionWiring(t *testing.T) {
	// Two calls per subagent: turn 0 = write tool call (StopToolUse), turn 1 =
	// final text (StopStop).
	prov := &scriptedProvider{scripts: [][]ai.AssistantMessageEvent{
		toolCallThenTextScript("write", "done"),
		textDoneScript("done"),
	}}
	perm := permission.New(permission.Options{Mode: permission.ModePlan, Enabled: true})
	writeTool := &fakeOrchTool{name: "write"}
	orch := New(Options{
		Provider:   prov,
		Model:      ai.Model{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "scripted-orch"},
		Tools:      []agent.AgentTool{writeTool},
		Permission: perm,
	})
	out, err := orch.RunSubtask(context.Background(), SubTask{ID: "1", Description: "do"})
	if err != nil {
		t.Fatalf("RunSubtask: %v", err)
	}
	// The final turn's text must be captured despite the write being blocked.
	if !strings.Contains(out, "done") {
		t.Fatalf("expected output to contain 'done', got %q", out)
	}
}

// TestToolFactoryIsUsedPerSubtask verifies that when ToolFactory is set, each
// subtask gets instances from the factory (defect 5). We count factory calls.
func TestToolFactoryIsUsedPerSubtask(t *testing.T) {
	prov := &scriptedProvider{scripts: [][]ai.AssistantMessageEvent{
		textDoneScript("result"),
	}}
	factoryCalls := 0
	orch := New(Options{
		Provider: prov,
		Model:    ai.Model{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "scripted-orch"},
		Tools:    []agent.AgentTool{&fakeOrchTool{name: "read"}},
		ToolFactory: func() []agent.AgentTool {
			factoryCalls++
			return []agent.AgentTool{&fakeOrchTool{name: "read"}}
		},
	})
	if _, err := orch.RunSubtask(context.Background(), SubTask{ID: "1", Description: "do"}); err != nil {
		t.Fatalf("RunSubtask: %v", err)
	}
	if factoryCalls != 1 {
		t.Fatalf("expected ToolFactory called once per subtask, got %d", factoryCalls)
	}
}

// TestToolsForSubtaskFallback verifies that without a factory, the shared Tools
// slice is used (backward compatibility, defect 5).
func TestToolsForSubtaskFallback(t *testing.T) {
	orch := New(Options{
		Provider: &scriptedProvider{},
		Model:    ai.Model{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "scripted-orch"},
		Tools:    []agent.AgentTool{&fakeOrchTool{name: "read"}},
	})
	got := orch.toolsForSubtask()
	if len(got) != 1 || got[0].Def().Name != "read" {
		t.Fatalf("expected fallback to shared Tools, got %+v", got)
	}
}

// TestParseSubTasksStripsCodeFence verifies markdown fence stripping (defect 3).
func TestParseSubTasksStripsCodeFence(t *testing.T) {
	fenced := "```json\n[{\"id\":\"1\",\"description\":\"x\"}]\n```"
	tasks, err := parseSubTasks(fenced)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(tasks) != 1 || tasks[0].ID != "1" {
		t.Fatalf("expected 1 task id 1, got %+v", tasks)
	}
}
