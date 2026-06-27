package harness

import (
	"context"
	"sync"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// mockProvider is a controllable Provider for tests. It returns scripted
// AssistantMessageEvent streams.
type mockProvider struct {
	mu       sync.Mutex
	models   []ai.Model
	scripts  [][]ai.AssistantMessageEvent // one script per call, popped in order
}

func newMockProvider(scripts ...[]ai.AssistantMessageEvent) *mockProvider {
	return &mockProvider{
		models:  []ai.Model{{ID: "mock-model", API: ai.APIAnthropicMessages, Provider: "mock"}},
		scripts: scripts,
	}
}

func (p *mockProvider) ID() string     { return "mock" }
func (p *mockProvider) Name() string   { return "mock" }
func (p *mockProvider) BaseURL() string { return "" }
func (p *mockProvider) Models() []ai.Model { return p.models }
func (p *mockProvider) GetModel(id string) (ai.Model, bool) {
	for _, m := range p.models {
		if m.ID == id {
			return m, true
		}
	}
	return ai.Model{}, false
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
