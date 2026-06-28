package agent

import (
	"context"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// LoopConfigBuilder constructs an AgentLoopConfig with defaults applied. It
// replaces direct struct-literal construction so defaults (ConvertToLlm,
// ToolExecution) live in one place and callers can't forget a required hook.
//
// Usage:
//
//	cfg := NewLoopConfig(model).
//	    WithConvertToLlm(myConverter).
//	    WithSteering(drainSteer).
//	    WithPermission(before, after).
//	    Build()
type LoopConfigBuilder struct {
	cfg AgentLoopConfig
}

// NewLoopConfig starts a builder with the given model. ConvertToLlm and
// ToolExecution defaults are applied at Build() time.
func NewLoopConfig(model ai.Model) *LoopConfigBuilder {
	return &LoopConfigBuilder{cfg: AgentLoopConfig{Model: model}}
}

// WithConvertToLlm sets the AgentMessage->Message converter.
func (b *LoopConfigBuilder) WithConvertToLlm(fn func([]AgentMessage) []ai.Message) *LoopConfigBuilder {
	b.cfg.ConvertToLlm = fn
	return b
}

// WithThinking sets the thinking level.
func (b *LoopConfigBuilder) WithThinking(level ai.ThinkingLevel) *LoopConfigBuilder {
	b.cfg.ThinkingLevel = level
	return b
}

// WithAPIKey sets the dynamic API key resolver.
func (b *LoopConfigBuilder) WithAPIKey(fn func(provider string) (string, error)) *LoopConfigBuilder {
	b.cfg.GetAPIKey = fn
	return b
}

// WithMaxTokens sets the max output tokens.
func (b *LoopConfigBuilder) WithMaxTokens(n int) *LoopConfigBuilder {
	b.cfg.MaxTokens = n
	return b
}

// WithTemperature sets the temperature.
func (b *LoopConfigBuilder) WithTemperature(t float64) *LoopConfigBuilder {
	b.cfg.Temperature = &t
	return b
}

// WithTransform sets the context-transform hook (pruning/injection).
func (b *LoopConfigBuilder) WithTransform(fn func([]AgentMessage) []AgentMessage) *LoopConfigBuilder {
	b.cfg.TransformContext = fn
	return b
}

// WithShouldStop sets the should-stop-after-turn hook.
func (b *LoopConfigBuilder) WithShouldStop(fn func(ShouldStopAfterTurnContext) bool) *LoopConfigBuilder {
	b.cfg.ShouldStopAfterTurn = fn
	return b
}

// WithPrepareNextTurn sets the prepare-next-turn hook.
func (b *LoopConfigBuilder) WithPrepareNextTurn(fn func(PrepareNextTurnContext) *AgentLoopTurnUpdate) *LoopConfigBuilder {
	b.cfg.PrepareNextTurn = fn
	return b
}

// WithSteering sets the steering-messages drain.
func (b *LoopConfigBuilder) WithSteering(fn func() []AgentMessage) *LoopConfigBuilder {
	b.cfg.GetSteeringMessages = fn
	return b
}

// WithFollowUp sets the follow-up-messages drain.
func (b *LoopConfigBuilder) WithFollowUp(fn func() []AgentMessage) *LoopConfigBuilder {
	b.cfg.GetFollowUpMessages = fn
	return b
}

// WithToolExecution sets the tool execution mode.
func (b *LoopConfigBuilder) WithToolExecution(mode ToolExecutionMode) *LoopConfigBuilder {
	b.cfg.ToolExecution = mode
	return b
}

// WithPermission sets the before/after tool-call hooks.
func (b *LoopConfigBuilder) WithPermission(
	before func(ctx context.Context, c BeforeToolCallContext) (*BeforeToolCallResult, error),
	after func(ctx context.Context, c AfterToolCallContext) (*AfterToolCallResult, error),
) *LoopConfigBuilder {
	b.cfg.BeforeToolCall = before
	b.cfg.AfterToolCall = after
	return b
}

// Build returns the configured AgentLoopConfig with defaults applied:
// ConvertToLlm defaults to DefaultConvertToLlm, ToolExecution to parallel.
func (b *LoopConfigBuilder) Build() AgentLoopConfig {
	if b.cfg.ConvertToLlm == nil {
		b.cfg.ConvertToLlm = DefaultConvertToLlm
	}
	if b.cfg.ToolExecution == "" {
		b.cfg.ToolExecution = ToolExecutionParallel
	}
	return b.cfg
}
