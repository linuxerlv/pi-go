package agent

import (
	"context"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// ToolExecutionMode controls how tool calls in a single assistant message are
// executed.
type ToolExecutionMode string

const (
	// ToolExecutionSequential executes each tool call fully before the next starts.
	ToolExecutionSequential ToolExecutionMode = "sequential"
	// ToolExecutionParallel prepares tool calls sequentially, then executes
	// allowed tools concurrently. tool_execution_end is emitted in completion
	// order; tool-result messages are emitted in assistant source order.
	ToolExecutionParallel ToolExecutionMode = "parallel"
)

// ThinkingLevel mirrors ai.ThinkingLevel for the agent layer.
type ThinkingLevel = ai.ThinkingLevel

// AgentMessage is a conversation message visible to the agent loop. It is
// either one of the base ai.Message types (UserMessage/AssistantMessage/
// ToolResultMessage) or a custom app message implementing AgentMessage.
type AgentMessage interface {
	ai.Message
}

// AgentContext is the snapshot passed into the agent loop.
type AgentContext struct {
	SystemPrompt string
	Messages     []AgentMessage
	Tools        []AgentTool
}

// AgentToolResult is the final or partial result produced by a tool.
type AgentToolResult struct {
	// Content is the text/image content returned to the model.
	Content []ai.ContentBlock
	// Details is arbitrary structured data for logs or UI rendering.
	Details any
	// Terminate hints the agent should stop after the current tool batch.
	// Early termination only happens when every finalized result in the batch
	// sets this to true.
	Terminate bool
}

// AgentToolUpdateCallback is used by tools to stream partial execution updates.
type AgentToolUpdateCallback func(partialResult AgentToolResult)

// AgentTool is a tool the agent runtime can invoke. It extends ai.Tool with
// execution behavior.
type AgentTool interface {
	// Def returns the tool definition (name/description/parameters).
	Def() ai.Tool
	// Label returns a human-readable label for UI display.
	Label() string
	// Execute runs the tool. Return an error to encode failure; do not encode
	// errors in Content. onUpdate streams partial results.
	Execute(ctx context.Context, toolCallID string, params map[string]any, onUpdate AgentToolUpdateCallback) (AgentToolResult, error)
	// ExecutionMode overrides the per-batch execution mode for this tool.
	// Return "" to use the loop default.
	ExecutionMode() ToolExecutionMode
}

// Preparer is an optional interface a tool may implement to shim raw tool-call
// arguments before schema validation.
type Preparer interface {
	PrepareArguments(args map[string]any) (map[string]any, error)
}

// BeforeToolCallResult is returned from BeforeToolCall to block execution.
type BeforeToolCallResult struct {
	Block  bool
	Reason string
}

// AfterToolCallResult partially overrides the executed tool result. Fields that
// are nil/keep their original values. No deep merge is performed.
type AfterToolCallResult struct {
	Content   *[]ai.ContentBlock
	Details   any
	IsError   *bool
	Terminate *bool
}

// BeforeToolCallContext is passed to BeforeToolCall.
type BeforeToolCallContext struct {
	AssistantMessage ai.AssistantMessage
	ToolCall         ai.ToolCall
	Args             any
	Context          AgentContext
}

// AfterToolCallContext is passed to AfterToolCall.
type AfterToolCallContext struct {
	AssistantMessage ai.AssistantMessage
	ToolCall         ai.ToolCall
	Args             any
	Result           AgentToolResult
	IsError          bool
	Context          AgentContext
}

// ShouldStopAfterTurnContext is passed to ShouldStopAfterTurn.
type ShouldStopAfterTurnContext struct {
	Message     ai.AssistantMessage
	ToolResults []ai.ToolResultMessage
	Context     AgentContext
	NewMessages []AgentMessage
}

// AgentLoopTurnUpdate replaces runtime state before the next provider request.
type AgentLoopTurnUpdate struct {
	Context       *AgentContext
	Model         *ai.Model
	ThinkingLevel *ThinkingLevel
}

// PrepareNextTurnContext is passed to PrepareNextTurn.
type PrepareNextTurnContext = ShouldStopAfterTurnContext

// AgentLoopConfig configures an agent loop run. It mirrors
// @earendil-works/pi-agent-core's AgentLoopConfig.
type AgentLoopConfig struct {
	Model        ai.Model
	ThinkingLevel ThinkingLevel
	APIKey       string
	MaxTokens    int
	Temperature  *float64

	// ConvertToLlm converts AgentMessage[] to LLM-compatible ai.Message[] before
	// each LLM call. Custom (non-LLM) messages should be filtered out here.
	// Required. Must not panic; return a safe fallback instead.
	ConvertToLlm func([]AgentMessage) []ai.Message

	// TransformContext is applied to AgentMessage[] before ConvertToLlm. Use it
	// for context-window management (pruning) or injecting external context.
	TransformContext func([]AgentMessage) []AgentMessage

	// GetAPIKey resolves an API key dynamically per LLM call (useful for
	// expiring OAuth tokens). Return "" to fall back to APIKey.
	GetAPIKey func(provider string) (string, error)

	// ShouldStopAfterTurn is called after each turn completes. If it returns
	// true, the loop emits agent_end and exits before polling steering/followup.
	ShouldStopAfterTurn func(ShouldStopAfterTurnContext) bool

	// PrepareNextTurn is called after turn_end and before the next turn. Return
	// a non-nil update to replace context/model/thinking for the next turn.
	PrepareNextTurn func(PrepareNextTurnContext) *AgentLoopTurnUpdate

	// GetSteeringMessages returns messages to inject mid-run (called between
	// turns). Return nil/empty for none.
	GetSteeringMessages func() []AgentMessage

	// GetFollowUpMessages returns messages to process after the agent would
	// otherwise stop. Return nil/empty for none.
	GetFollowUpMessages func() []AgentMessage

	// ToolExecution is the default tool execution mode. Default: parallel.
	ToolExecution ToolExecutionMode

	// BeforeToolCall is called before a tool executes, after argument
	// validation. Return Block=true to prevent execution.
	BeforeToolCall func(ctx context.Context, c BeforeToolCallContext) (*BeforeToolCallResult, error)

	// AfterToolCall is called after a tool finishes, before tool_execution_end.
	// Return a non-nil result to override parts of the executed result.
	AfterToolCall func(ctx context.Context, c AfterToolCallContext) (*AfterToolCallResult, error)
}

// AgentEvent is emitted by the agent loop for UI updates. Mirrors the
// @earendil-works/pi-agent-core AgentEvent union.
type AgentEvent interface {
	agentEvent()
	EventType() string
}

// AgentStartEvent begins a run.
type AgentStartEvent struct{}

// AgentEndEvent ends a run with the messages produced this run.
type AgentEndEvent struct {
	Messages []AgentMessage
}

// TurnStartEvent begins a turn (one assistant response + tool calls/results).
type TurnStartEvent struct{}

// TurnEndEvent ends a turn.
type TurnEndEvent struct {
	Message     AgentMessage
	ToolResults []ai.ToolResultMessage
}

// MessageStartEvent is emitted for user/assistant/toolResult messages.
type MessageStartEvent struct {
	Message AgentMessage
}

// MessageUpdateEvent is emitted only for assistant messages during streaming.
type MessageUpdateEvent struct {
	Message               AgentMessage
	AssistantMessageEvent ai.AssistantMessageEvent
}

// MessageEndEvent is emitted when a message is finalized.
type MessageEndEvent struct {
	Message AgentMessage
}

// ToolExecutionStartEvent begins a tool execution.
type ToolExecutionStartEvent struct {
	ToolCallID string
	ToolName   string
	Args       any
}

// ToolExecutionUpdateEvent carries a partial tool result.
type ToolExecutionUpdateEvent struct {
	ToolCallID   string
	ToolName     string
	Args         any
	PartialResult AgentToolResult
}

// ToolExecutionEndEvent ends a tool execution.
type ToolExecutionEndEvent struct {
	ToolCallID string
	ToolName   string
	Result     AgentToolResult
	IsError    bool
}

func (AgentStartEvent) agentEvent()        {}
func (AgentEndEvent) agentEvent()          {}
func (TurnStartEvent) agentEvent()         {}
func (TurnEndEvent) agentEvent()           {}
func (MessageStartEvent) agentEvent()      {}
func (MessageUpdateEvent) agentEvent()     {}
func (MessageEndEvent) agentEvent()        {}
func (ToolExecutionStartEvent) agentEvent() {}
func (ToolExecutionUpdateEvent) agentEvent() {}
func (ToolExecutionEndEvent) agentEvent()   {}

func (AgentStartEvent) EventType() string        { return "agent_start" }
func (AgentEndEvent) EventType() string          { return "agent_end" }
func (TurnStartEvent) EventType() string         { return "turn_start" }
func (TurnEndEvent) EventType() string           { return "turn_end" }
func (MessageStartEvent) EventType() string      { return "message_start" }
func (MessageUpdateEvent) EventType() string     { return "message_update" }
func (MessageEndEvent) EventType() string        { return "message_end" }
func (ToolExecutionStartEvent) EventType() string { return "tool_execution_start" }
func (ToolExecutionUpdateEvent) EventType() string { return "tool_execution_update" }
func (ToolExecutionEndEvent) EventType() string   { return "tool_execution_end" }

// EventSink is the callback the loop emits events to.
type EventSink func(AgentEvent) error
