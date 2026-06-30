package agent

import (
	"testing"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// TestDefaultConvertToLlmPassesThrough confirms the converter returns every
// AgentMessage as an ai.Message. (AgentMessage embeds ai.Message, so the
// type-assertion filter is effectively a pass-through in practice.)
func TestDefaultConvertToLlmPassesThrough(t *testing.T) {
	in := []AgentMessage{
		ai.UserMessage{Content: "hi", Timestamp: 1},
		ai.AssistantMessage{Provider: "mock", Model: "m", Timestamp: 2},
		ai.ToolResultMessage{ToolCallID: "c1", ToolName: "read", Timestamp: 3},
	}
	out := DefaultConvertToLlm(in)
	if len(out) != 3 {
		t.Fatalf("expected 3 messages through, got %d", len(out))
	}
	if out[0].Role() != "user" || out[1].Role() != "assistant" || out[2].Role() != "toolResult" {
		t.Fatalf("roles not preserved: %v %v %v", out[0].Role(), out[1].Role(), out[2].Role())
	}
}

// TestLoopConfigBuilderDefaults verifies Build() applies the documented defaults
// (ConvertToLlm=DefaultConvertToLlm, ToolExecution=parallel) when unset, and
// preserves explicitly-set values.
func TestLoopConfigBuilderDefaults(t *testing.T) {
	cfg := NewLoopConfig(ai.Model{ID: "m"}).Build()
	if cfg.ConvertToLlm == nil {
		t.Fatal("ConvertToLlm should default to non-nil")
	}
	if cfg.ToolExecution != ToolExecutionParallel {
		t.Fatalf("ToolExecution should default to parallel, got %s", cfg.ToolExecution)
	}
	if cfg.Model.ID != "m" {
		t.Fatalf("model not preserved: %+v", cfg.Model)
	}

	custom := func([]AgentMessage) []ai.Message { return nil }
	cfg2 := NewLoopConfig(ai.Model{ID: "m"}).
		WithConvertToLlm(custom).
		WithToolExecution(ToolExecutionSequential).
		WithMaxTokens(123).
		Build()
	if cfg2.ConvertToLlm == nil || cfg2.ToolExecution != ToolExecutionSequential {
		t.Fatalf("explicit values not preserved: %+v", cfg2)
	}
	if cfg2.MaxTokens != 123 {
		t.Fatalf("MaxTokens not preserved: %d", cfg2.MaxTokens)
	}
}
