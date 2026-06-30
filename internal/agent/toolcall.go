package agent

import (
	"context"
	"sync"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// executedToolCallBatch is the result of executing a batch of tool calls.
type executedToolCallBatch struct {
	messages  []ai.ToolResultMessage
	terminate bool
}

type finalizedToolCall struct {
	toolCall ai.ToolCall
	result   AgentToolResult
	isError  bool
}

// executeToolCalls dispatches to sequential or parallel execution. Mirrors
// agent-loop.ts:373. If the config mode is sequential OR any tool in the batch
// declares executionMode "sequential", the whole batch runs sequentially.
func executeToolCalls(
	ctx context.Context,
	currentContext AgentContext,
	assistantMessage ai.AssistantMessage,
	toolCalls []ai.ToolCall,
	config AgentLoopConfig,
	emit EventSink,
) (executedToolCallBatch, error) {
	hasSequential := false
	for _, tc := range toolCalls {
		if t := findTool(currentContext.Tools, tc.Name); t != nil && t.ExecutionMode() == ToolExecutionSequential {
			hasSequential = true
			break
		}
	}
	if config.ToolExecution == ToolExecutionSequential || hasSequential {
		return executeToolCallsSequential(ctx, currentContext, assistantMessage, toolCalls, config, emit)
	}
	return executeToolCallsParallel(ctx, currentContext, assistantMessage, toolCalls, config, emit)
}

// prepareAndExecuteOne runs the prepare->execute->finalize pipeline for a single
// tool call, returning the finalized outcome. It is the shared core of both the
// sequential and parallel executors (which differ only in scheduling and emit
// ordering). Emit of start/end events is the caller's responsibility, so each
// executor controls its own ordering semantics. onUpdate is the emit sink for
// tool_execution_update events produced during execution (parallel wraps it in
// a mutex).
func prepareAndExecuteOne(
	ctx context.Context,
	currentContext AgentContext,
	assistantMessage ai.AssistantMessage,
	tc ai.ToolCall,
	config AgentLoopConfig,
	onUpdate EventSink,
) (finalizedToolCall, error) {
	prep, err := prepareToolCall(ctx, currentContext, assistantMessage, tc, config)
	if err != nil {
		return finalizedToolCall{}, err
	}
	if prep.immediate {
		return finalizedToolCall{toolCall: tc, result: prep.result, isError: prep.isError}, nil
	}
	executedResult, executedIsError, err := executePreparedToolCall(ctx, prep, onUpdate)
	if err != nil {
		return finalizedToolCall{}, err
	}
	f, err := finalizeExecutedToolCall(ctx, currentContext, assistantMessage, prep, executedResult, executedIsError, config)
	if err != nil {
		return finalizedToolCall{}, err
	}
	return f, nil
}

func executeToolCallsSequential(
	ctx context.Context,
	currentContext AgentContext,
	assistantMessage ai.AssistantMessage,
	toolCalls []ai.ToolCall,
	config AgentLoopConfig,
	emit EventSink,
) (executedToolCallBatch, error) {
	var finalized []finalizedToolCall
	messages := make([]ai.ToolResultMessage, 0, len(toolCalls))

	for _, tc := range toolCalls {
		emit(ToolExecutionStartEvent{ToolCallID: tc.ID, ToolName: tc.Name, Args: tc.Arguments})

		f, err := prepareAndExecuteOne(ctx, currentContext, assistantMessage, tc, config, emit)
		if err != nil {
			return executedToolCallBatch{}, err
		}

		emit(ToolExecutionEndEvent{ToolCallID: f.toolCall.ID, ToolName: f.toolCall.Name, Result: f.result, IsError: f.isError})
		trm := createToolResultMessage(f)
		emit(MessageStartEvent{Message: trm})
		emit(MessageEndEvent{Message: trm})
		finalized = append(finalized, f)
		messages = append(messages, trm)

		if ctx.Err() != nil {
			break
		}
	}

	return executedToolCallBatch{
		messages:  messages,
		terminate: shouldTerminateBatch(finalized),
	}, nil
}

func executeToolCallsParallel(
	ctx context.Context,
	currentContext AgentContext,
	assistantMessage ai.AssistantMessage,
	toolCalls []ai.ToolCall,
	config AgentLoopConfig,
	emit EventSink,
) (executedToolCallBatch, error) {
	type preparedEntry struct {
		index      int
		toolCall   ai.ToolCall
		prepared   *preparedToolCall
		immediate  bool
		immResult  AgentToolResult
		immIsError bool
	}

	// Prepare sequentially (validation + beforeToolCall must be serial).
	prepared := make([]preparedEntry, 0, len(toolCalls))
	for i, tc := range toolCalls {
		emit(ToolExecutionStartEvent{ToolCallID: tc.ID, ToolName: tc.Name, Args: tc.Arguments})
		prep, err := prepareToolCall(ctx, currentContext, assistantMessage, tc, config)
		if err != nil {
			return executedToolCallBatch{}, err
		}
		entry := preparedEntry{index: i, toolCall: tc}
		if prep.immediate {
			entry.immediate = true
			entry.immResult = prep.result
			entry.immIsError = prep.isError
		} else {
			entry.prepared = prep
		}
		prepared = append(prepared, entry)
		if ctx.Err() != nil {
			break
		}
	}

	// Execute allowed tools concurrently. Finalized results land at their
	// source index; tool_execution_end is emitted in completion order.
	finalized := make([]finalizedToolCall, len(prepared))
	for i := range finalized {
		// Pre-fill immediate entries so they exist regardless of completion.
		if prepared[i].immediate {
			finalized[i] = finalizedToolCall{
				toolCall: prepared[i].toolCall,
				result:   prepared[i].immResult,
				isError:  prepared[i].immIsError,
			}
		}
	}

	var wg sync.WaitGroup
	var mu sync.Mutex // guards emit and finalized writes from concurrent tools

	for i, entry := range prepared {
		if entry.immediate || entry.prepared == nil {
			// Immediate outcomes: emit end now, in source order.
			f := finalized[i]
			emit(ToolExecutionEndEvent{ToolCallID: f.toolCall.ID, ToolName: f.toolCall.Name, Result: f.result, IsError: f.isError})
			continue
		}
		wg.Add(1)
		go func(i int, entry preparedEntry) {
			defer wg.Done()
			executedResult, executedIsError, err := executePreparedToolCall(ctx, entry.prepared, func(e AgentEvent) error {
				mu.Lock()
				defer mu.Unlock()
				return emit(e)
			})
			if err != nil {
				// Execution errors are non-fatal to the loop; encode as error result.
				f := finalizedToolCall{
					toolCall: entry.toolCall,
					result:   errorToolResult(err.Error()),
					isError:  true,
				}
				mu.Lock()
				finalized[i] = f
				emit(ToolExecutionEndEvent{ToolCallID: f.toolCall.ID, ToolName: f.toolCall.Name, Result: f.result, IsError: true})
				mu.Unlock()
				return
			}
			// Run the AfterToolCall hook OUTSIDE mu: it is user code that may be
			// slow or block, and holding mu here would serialize all concurrent
			// tools. finalize only transforms the result (no emit), so the lock
			// is not needed for it.
			f, _ := finalizeExecutedToolCall(ctx, currentContext, assistantMessage, entry.prepared, executedResult, executedIsError, config)
			mu.Lock()
			finalized[i] = f
			emit(ToolExecutionEndEvent{ToolCallID: f.toolCall.ID, ToolName: f.toolCall.Name, Result: f.result, IsError: f.isError})
			mu.Unlock()
		}(i, entry)
	}
	wg.Wait()

	// Emit tool-result messages in assistant source order.
	messages := make([]ai.ToolResultMessage, 0, len(finalized))
	for _, f := range finalized {
		trm := createToolResultMessage(f)
		emit(MessageStartEvent{Message: trm})
		emit(MessageEndEvent{Message: trm})
		messages = append(messages, trm)
	}

	return executedToolCallBatch{
		messages:  messages,
		terminate: shouldTerminateBatch(finalized),
	}, nil
}

// preparedToolCall is a validated, about-to-execute tool call. immediate marks
// outcomes that bypass execution (not-found / validation error / blocked /
// aborted), carrying a ready-made error result.
type preparedToolCall struct {
	immediate bool
	toolCall  ai.ToolCall
	tool      AgentTool
	args      map[string]any
	result    AgentToolResult // only when immediate
	isError   bool            // only when immediate
}

func prepareToolCall(
	ctx context.Context,
	currentContext AgentContext,
	assistantMessage ai.AssistantMessage,
	toolCall ai.ToolCall,
	config AgentLoopConfig,
) (*preparedToolCall, error) {
	tool := findTool(currentContext.Tools, toolCall.Name)
	if tool == nil {
		return &preparedToolCall{
			immediate: true,
			toolCall:  toolCall,
			result:    errorToolResult("Tool " + toolCall.Name + " not found"),
			isError:   true,
		}, nil
	}

	args := toolCall.Arguments
	if prep, ok := tool.(Preparer); ok {
		prepared, err := prep.PrepareArguments(args)
		if err != nil {
			return &preparedToolCall{
				immediate: true,
				toolCall:  toolCall,
				result:    errorToolResult(err.Error()),
				isError:   true,
			}, nil
		}
		args = prepared
	}

	// Validate arguments against the tool's schema.
	validated, err := ai.ValidateToolArguments(tool.Def(), ai.ToolCall{
		Type:      toolCall.Type,
		ID:        toolCall.ID,
		Name:      toolCall.Name,
		Arguments: args,
	})
	if err != nil {
		return &preparedToolCall{
			immediate: true,
			toolCall:  toolCall,
			result:    errorToolResult(err.Error()),
			isError:   true,
		}, nil
	}

	if config.BeforeToolCall != nil {
		before, err := config.BeforeToolCall(ctx, BeforeToolCallContext{
			AssistantMessage: assistantMessage,
			ToolCall:         toolCall,
			Args:             validated,
			Context:          currentContext,
		})
		if err != nil {
			return &preparedToolCall{
				immediate: true,
				toolCall:  toolCall,
				result:    errorToolResult(err.Error()),
				isError:   true,
			}, nil
		}
		if ctx.Err() != nil {
			return &preparedToolCall{
				immediate: true,
				toolCall:  toolCall,
				result:    errorToolResult("Operation aborted"),
				isError:   true,
			}, nil
		}
		if before != nil && before.Block {
			reason := "Tool execution was blocked"
			if before.Reason != "" {
				reason = before.Reason
			}
			return &preparedToolCall{
				immediate: true,
				toolCall:  toolCall,
				result:    errorToolResult(reason),
				isError:   true,
			}, nil
		}
	}
	if ctx.Err() != nil {
		return &preparedToolCall{
			immediate: true,
			toolCall:  toolCall,
			result:    errorToolResult("Operation aborted"),
			isError:   true,
		}, nil
	}

	return &preparedToolCall{
		toolCall: toolCall,
		tool:     tool,
		args:     validated,
	}, nil
}

// executePreparedToolCall runs the tool, forwarding onUpdate events. Errors are
// converted to error results (isError=true). Mirrors agent-loop.ts:628.
func executePreparedToolCall(
	ctx context.Context,
	prep *preparedToolCall,
	emit EventSink,
) (AgentToolResult, bool, error) {
	result, err := prep.tool.Execute(ctx, prep.toolCall.ID, prep.args, func(partial AgentToolResult) {
		emit(ToolExecutionUpdateEvent{
			ToolCallID:    prep.toolCall.ID,
			ToolName:      prep.toolCall.Name,
			Args:          prep.toolCall.Arguments,
			PartialResult: partial,
		})
	})
	if err != nil {
		return errorToolResult(err.Error()), true, nil
	}
	return result, false, nil
}

// finalizeExecutedToolCall applies the AfterToolCall override. Mirrors
// agent-loop.ts:671.
func finalizeExecutedToolCall(
	ctx context.Context,
	currentContext AgentContext,
	assistantMessage ai.AssistantMessage,
	prep *preparedToolCall,
	executedResult AgentToolResult,
	executedIsError bool,
	config AgentLoopConfig,
) (finalizedToolCall, error) {
	result := executedResult
	isError := executedIsError

	if config.AfterToolCall != nil {
		after, err := config.AfterToolCall(ctx, AfterToolCallContext{
			AssistantMessage: assistantMessage,
			ToolCall:         prep.toolCall,
			Args:             prep.args,
			Result:           result,
			IsError:          isError,
			Context:          currentContext,
		})
		if err != nil {
			result = errorToolResult(err.Error())
			isError = true
		} else if after != nil {
			if after.Content != nil {
				result.Content = *after.Content
			}
			if after.Details != nil {
				result.Details = after.Details
			}
			if after.IsError != nil {
				isError = *after.IsError
			}
			if after.Terminate != nil {
				result.Terminate = *after.Terminate
			}
		}
	}

	return finalizedToolCall{
		toolCall: prep.toolCall,
		result:   result,
		isError:  isError,
	}, nil
}

// shouldTerminateBatch returns true only when every finalized result in the
// batch has Terminate=true (and the batch is non-empty). Mirrors agent-loop.ts:544.
func shouldTerminateBatch(finalized []finalizedToolCall) bool {
	if len(finalized) == 0 {
		return false
	}
	for _, f := range finalized {
		if !f.result.Terminate {
			return false
		}
	}
	return true
}

func createToolResultMessage(f finalizedToolCall) ai.ToolResultMessage {
	return ai.ToolResultMessage{
		ToolCallID: f.toolCall.ID,
		ToolName:   f.toolCall.Name,
		Content:    f.result.Content,
		Details:    f.result.Details,
		IsError:    f.isError,
		Timestamp:  ai.Now(),
	}
}

func errorToolResult(message string) AgentToolResult {
	return AgentToolResult{
		Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: message}},
		Details: map[string]any{},
	}
}

func findTool(tools []AgentTool, name string) AgentTool {
	for _, t := range tools {
		if t.Def().Name == name {
			return t
		}
	}
	return nil
}
