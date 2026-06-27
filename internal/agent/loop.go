// Package agent implements the pi-agent-core agent loop: a Go port of
// @earendil-works/pi-agent-core's agent-loop.ts. The loop calls an LLM via a
// provider, parses tool calls, executes them, feeds results back, and repeats
// until the model stops calling tools.
package agent

import (
	"context"
	"fmt"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// RunAgentLoop starts an agent loop with new prompt messages. It is the Go
// counterpart of pi-agent-core's runAgentLoop: prompts are added to the context
// and events are emitted via emit as the run progresses. Returns the new
// messages produced during this run.
func RunAgentLoop(
	ctx context.Context,
	prompts []AgentMessage,
	context_ AgentContext,
	config AgentLoopConfig,
	provider ai.Provider,
	emit EventSink,
) ([]AgentMessage, error) {
	if config.ConvertToLlm == nil {
		config.ConvertToLlm = DefaultConvertToLlm
	}
	if config.ToolExecution == "" {
		config.ToolExecution = ToolExecutionParallel
	}

	newMessages := append([]AgentMessage(nil), prompts...)
	currentContext := AgentContext{
		SystemPrompt: context_.SystemPrompt,
		Tools:        context_.Tools,
		Messages:     append(append([]AgentMessage(nil), context_.Messages...), prompts...),
	}

	emit(AgentStartEvent{})
	emit(TurnStartEvent{})
	for _, p := range prompts {
		emit(MessageStartEvent{Message: p})
		emit(MessageEndEvent{Message: p})
	}

	if err := runLoop(ctx, currentContext, &newMessages, config, provider, emit); err != nil {
		return newMessages, err
	}
	return newMessages, nil
}

// runLoop is the main loop shared by RunAgentLoop. Mirrors pi-agent-core's
// runLoop (agent-loop.ts:155).
func runLoop(
	ctx context.Context,
	initialContext AgentContext,
	newMessages *[]AgentMessage,
	initialConfig AgentLoopConfig,
	provider ai.Provider,
	emit EventSink,
) error {
	currentContext := initialContext
	config := initialConfig
	firstTurn := true

	// Check for steering messages at start (user may have queued while waiting).
	pendingMessages := callSteering(config)

	// Outer loop: continues when follow-up messages arrive after the agent
	// would otherwise stop.
	for {
		hasMoreToolCalls := true

		// Inner loop: process tool calls and steering messages.
		for hasMoreToolCalls || len(pendingMessages) > 0 {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if !firstTurn {
				emit(TurnStartEvent{})
			} else {
				firstTurn = false
			}

			// Inject pending messages (steering / initial prompt).
			if len(pendingMessages) > 0 {
				for _, m := range pendingMessages {
					emit(MessageStartEvent{Message: m})
					emit(MessageEndEvent{Message: m})
				currentContext.Messages = append(currentContext.Messages, m)
				*newMessages = append(*newMessages, m)
				}
				pendingMessages = nil
			}

			// Stream assistant response.
			message, err := streamAssistantResponse(ctx, currentContext, config, provider, emit)
			if err != nil {
				return err
			}
			*newMessages = append(*newMessages, message)
			// The assistant message must be visible to the next turn's LLM
			// call (especially when it carries tool_use blocks that the
			// following tool_result messages reference by id). agent-loop.ts
			// relies on JS object-reference mutation; in Go we append to the
			// current context explicitly since streamAssistantResponse receives
			// AgentContext by value.
			currentContext.Messages = append(currentContext.Messages, message)

			if message.StopReason == ai.StopError || message.StopReason == ai.StopAborted {
				emit(TurnEndEvent{Message: message, ToolResults: nil})
				emit(AgentEndEvent{Messages: *newMessages})
				return nil
			}

			// Check for tool calls.
			var toolCalls []ai.ToolCall
			for _, c := range message.Content {
				if tc, ok := c.(ai.ToolCall); ok {
					toolCalls = append(toolCalls, tc)
				}
			}

			var toolResults []ai.ToolResultMessage
			hasMoreToolCalls = false
			if len(toolCalls) > 0 {
				batch, err := executeToolCalls(ctx, currentContext, message, toolCalls, config, emit)
				if err != nil {
					return err
				}
				toolResults = batch.messages
				hasMoreToolCalls = !batch.terminate
				for _, r := range toolResults {
					currentContext.Messages = append(currentContext.Messages, r)
					*newMessages = append(*newMessages, r)
				}
			}

			emit(TurnEndEvent{Message: message, ToolResults: toolResults})

			// prepareNextTurn: optionally replace context/model/thinking.
			if config.PrepareNextTurn != nil {
				ntCtx := PrepareNextTurnContext{
					Message:     message,
					ToolResults: toolResults,
					Context:     currentContext,
					NewMessages: *newMessages,
				}
				if upd := config.PrepareNextTurn(ntCtx); upd != nil {
					if upd.Context != nil {
						currentContext = *upd.Context
					}
					if upd.Model != nil {
						config.Model = *upd.Model
					}
					if upd.ThinkingLevel != nil {
						config.ThinkingLevel = *upd.ThinkingLevel
					}
				}
			}

			// shouldStopAfterTurn: graceful stop before steering/followup.
			if config.ShouldStopAfterTurn != nil {
			if config.ShouldStopAfterTurn(ShouldStopAfterTurnContext{
				Message:     message,
				ToolResults: toolResults,
				Context:     currentContext,
				NewMessages: *newMessages,
			}) {
				emit(AgentEndEvent{Messages: *newMessages})
				return nil
			}
			}

			pendingMessages = callSteering(config)
		}

		// Agent would stop here. Check for follow-up messages.
		var followUp []AgentMessage
		if config.GetFollowUpMessages != nil {
			followUp = config.GetFollowUpMessages()
		}
		if len(followUp) > 0 {
			pendingMessages = followUp
			continue
		}
		break
	}

	emit(AgentEndEvent{Messages: *newMessages})
	return nil
}

func callSteering(config AgentLoopConfig) []AgentMessage {
	if config.GetSteeringMessages == nil {
		return nil
	}
	return config.GetSteeringMessages()
}

// streamAssistantResponse streams an assistant response from the LLM, emitting
// message_start/update/end events. Mirrors agent-loop.ts:275.
func streamAssistantResponse(
	ctx context.Context,
	context_ AgentContext,
	config AgentLoopConfig,
	provider ai.Provider,
	emit EventSink,
) (ai.AssistantMessage, error) {
	messages := context_.Messages
	if config.TransformContext != nil {
		messages = config.TransformContext(messages)
	}

	llmMessages := config.ConvertToLlm(messages)

	llmContext := ai.Context{
		SystemPrompt: context_.SystemPrompt,
		Messages:     llmMessages,
		Tools:        toolsToAI(context_.Tools),
	}

	options := &ai.SimpleStreamOptions{
		StreamOptions: ai.StreamOptions{
			APIKey:    config.APIKey,
			MaxTokens: config.MaxTokens,
		},
		Reasoning: config.ThinkingLevel,
	}
	if config.Temperature != nil {
		options.Temperature = config.Temperature
	}
	if config.GetAPIKey != nil {
		if k, err := config.GetAPIKey(config.Model.Provider); err == nil && k != "" {
			options.APIKey = k
		}
	}

	stream := provider.StreamSimple(ctx, config.Model, llmContext, options)

	var partial *ai.AssistantMessage
	addedPartial := false

	for ev := range stream.Range {
		switch e := ev.(type) {
		case ai.StartEvent:
			m := e.Partial
			partial = &m
			context_.Messages = append(context_.Messages, m)
			addedPartial = true
			emit(MessageStartEvent{Message: m})

		case ai.TextStartEvent, ai.TextDeltaEvent, ai.TextEndEvent,
			ai.ThinkingStartEvent, ai.ThinkingDeltaEvent, ai.ThinkingEndEvent,
			ai.ToolCallStartEvent, ai.ToolCallDeltaEvent, ai.ToolCallEndEvent:
			if partial != nil {
				m := partialFromEvent(e, *partial)
				partial = &m
				// Replace the last message (the in-flight partial).
				idx := len(context_.Messages) - 1
				if idx >= 0 {
					context_.Messages[idx] = m
				}
				emit(MessageUpdateEvent{Message: m, AssistantMessageEvent: e})
			}

		case ai.DoneEvent, ai.ErrorEvent:
			final, _ := ai.TerminalMessage(e)
			if addedPartial {
				context_.Messages[len(context_.Messages)-1] = final
			} else {
				context_.Messages = append(context_.Messages, final)
				emit(MessageStartEvent{Message: final})
			}
			emit(MessageEndEvent{Message: final})
			return final, nil
		}
	}

	// Stream ended without a terminal event (shouldn't normally happen); use
	// the last partial or synthesize an error.
	if partial != nil {
		emit(MessageEndEvent{Message: *partial})
		return *partial, nil
	}
	err := fmt.Errorf("stream ended without a terminal event")
	final := ai.AssistantMessage{
		Content:      []ai.ContentBlock{ai.TextContent{Type: "text", Text: ""}},
		API:          config.Model.API,
		Provider:     config.Model.Provider,
		Model:        config.Model.ID,
		StopReason:   ai.StopError,
		ErrorMessage: err.Error(),
		Timestamp:    ai.Now(),
	}
	emit(MessageStartEvent{Message: final})
	emit(MessageEndEvent{Message: final})
	return final, nil
}

// partialFromEvent extracts the partial AssistantMessage carried by a
// non-terminal AssistantMessageEvent.
func partialFromEvent(e ai.AssistantMessageEvent, fallback ai.AssistantMessage) ai.AssistantMessage {
	switch ev := e.(type) {
	case ai.TextStartEvent:
		return ev.Partial
	case ai.TextDeltaEvent:
		return ev.Partial
	case ai.TextEndEvent:
		return ev.Partial
	case ai.ThinkingStartEvent:
		return ev.Partial
	case ai.ThinkingDeltaEvent:
		return ev.Partial
	case ai.ThinkingEndEvent:
		return ev.Partial
	case ai.ToolCallStartEvent:
		return ev.Partial
	case ai.ToolCallDeltaEvent:
		return ev.Partial
	case ai.ToolCallEndEvent:
		return ev.Partial
	}
	return fallback
}

func toolsToAI(tools []AgentTool) []ai.Tool {
	out := make([]ai.Tool, 0, len(tools))
	for _, t := range tools {
		out = append(out, t.Def())
	}
	return out
}
