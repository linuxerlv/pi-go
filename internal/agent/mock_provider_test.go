package agent

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// mockProvider is a controllable provider for agent-loop tests. It returns
// scripted AssistantMessageEvent streams, one per StreamSimple call.
type mockProvider struct {
	mu      sync.Mutex
	scripts [][]ai.AssistantMessageEvent
}

func newMockProvider(scripts ...[]ai.AssistantMessageEvent) *mockProvider {
	return &mockProvider{scripts: scripts}
}

func (p *mockProvider) ID() string      { return "mock" }
func (p *mockProvider) Name() string    { return "mock" }
func (p *mockProvider) BaseURL() string { return "" }
func (p *mockProvider) Models() []ai.Model {
	return []ai.Model{{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "mock"}}
}
func (p *mockProvider) GetModel(id string) (ai.Model, bool) {
	return ai.Model{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "mock"}, true
}

func (p *mockProvider) StreamSimple(ctx context.Context, model ai.Model, context ai.Context, options *ai.SimpleStreamOptions) ai.AssistantMessageEventStream {
	stream := ai.NewAssistantMessageEventStream()
	p.mu.Lock()
	var script []ai.AssistantMessageEvent
	if len(p.scripts) > 0 {
		script = p.scripts[0]
		p.scripts = p.scripts[1:]
	}
	p.mu.Unlock()
	go func() {
		defer stream.End(nil)
		for _, ev := range script {
			stream.Push(ev)
		}
	}()
	return stream
}

// scriptTextTurn builds a script for one assistant turn emitting text + done.
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
		ai.TextDeltaEvent{ContentIndex: 0, Delta: text, Partial: msg},
		ai.DoneEvent{Reason: ai.StopStop, Message: msg},
	}
}

// scriptToolCallTurn builds a script for an assistant turn that emits a tool
// call (stopReason=toolUse).
func scriptToolCallTurn(toolID, toolName string, args map[string]any) []ai.AssistantMessageEvent {
	argsJSON, _ := json.Marshal(args)
	tc := ai.ToolCall{Type: "toolCall", ID: toolID, Name: toolName, Arguments: args, ArgumentsRaw: argsJSON}
	msg := ai.AssistantMessage{
		Content:    []ai.ContentBlock{tc},
		API:        ai.APIAnthropicMessages,
		Provider:   "mock",
		Model:      "mock-model",
		StopReason: ai.StopToolUse,
		Timestamp:  ai.Now(),
	}
	return []ai.AssistantMessageEvent{
		ai.StartEvent{Partial: msg},
		ai.ToolCallStartEvent{ContentIndex: 0, Partial: msg},
		ai.ToolCallEndEvent{ContentIndex: 0, ToolCall: tc, Partial: msg},
		ai.DoneEvent{Reason: ai.StopToolUse, Message: msg},
	}
}

func scriptErrorTurn(msg string) []ai.AssistantMessageEvent {
	m := ai.AssistantMessage{
		Content:      []ai.ContentBlock{ai.TextContent{Type: "text", Text: ""}},
		API:          ai.APIAnthropicMessages,
		Provider:     "mock",
		Model:        "mock-model",
		StopReason:   ai.StopError,
		ErrorMessage: msg,
		Timestamp:    ai.Now(),
	}
	return []ai.AssistantMessageEvent{ai.StartEvent{Partial: m}, ai.ErrorEvent{Reason: ai.StopError, Error: m}}
}
