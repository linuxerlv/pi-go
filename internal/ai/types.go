// Package ai is the unified multi-provider LLM abstraction layer, a Go port of
// @earendil-works/pi-ai. It defines the provider-agnostic types (Context,
// Message, content blocks, Usage, Model, Tool), the streaming event protocol
// (AssistantMessageEvent), and the EventStream primitive. Provider adapters
// under internal/ai/provider translate vendor SDK event streams into this
// protocol; the agent loop consumes only streamSimple.
package ai

import (
	"encoding/json"
	"time"
)

// StopReason is the terminal reason for an assistant message.
type StopReason string

const (
	StopStop    StopReason = "stop"
	StopLength  StopReason = "length"
	StopToolUse StopReason = "toolUse"
	StopError   StopReason = "error"
	StopAborted StopReason = "aborted"
)

// Api is an API identifier. Known values are listed below; custom APIs are
// allowed via arbitrary strings.
type Api string

const (
	APIAnthropicMessages    Api = "anthropic-messages"
	APIOpenAICompletions    Api = "openai-completions"
	APIOpenAIResponses      Api = "openai-responses"
	APIGoogleGenerativeAI   Api = "google-generative-ai"
	APIGoogleVertex         Api = "google-vertex"
	APIMistralConversations Api = "mistral-conversations"
	APIBedrockConverse      Api = "bedrock-converse-stream"
)

// ContentBlock is one block of a message's content. The concrete types are
// TextContent, ThinkingContent, ImageContent, and ToolCall.
type ContentBlock interface {
	contentBlock()
}

// TextContent is a plain text block.
type TextContent struct {
	Type         string `json:"type"`
	Text         string `json:"text"`
	TextSignature string `json:"textSignature,omitempty"`
}

// ThinkingContent is a reasoning/thinking block.
type ThinkingContent struct {
	Type              string `json:"type"`
	Thinking          string `json:"thinking"`
	ThinkingSignature string `json:"thinkingSignature,omitempty"`
	Redacted          bool   `json:"redacted,omitempty"`
}

// ImageContent is a base64-encoded image block.
type ImageContent struct {
	Type     string `json:"type"`
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
}

// ToolCall is a tool invocation requested by the assistant. Arguments is the
// decoded JSON object; raw JSON is kept in ArgumentsRaw for round-tripping.
type ToolCall struct {
	Type           string         `json:"type"`
	ID             string         `json:"id"`
	Name           string         `json:"name"`
	Arguments      map[string]any `json:"-"`
	ArgumentsRaw   json.RawMessage `json:"arguments,omitempty"`
	ThoughtSignature string       `json:"thoughtSignature,omitempty"`
}

func (TextContent) contentBlock()     {}
func (ThinkingContent) contentBlock() {}
func (ImageContent) contentBlock()    {}
func (ToolCall) contentBlock()        {}

// Message is a conversation message: UserMessage, AssistantMessage, or
// ToolResultMessage.
type Message interface {
	message()
	Role() string
}

// UserMessage is a message from the user. Content is either a string or a
// []ContentBlock (text/image).
type UserMessage struct {
	Timestamp int64 `json:"timestamp"`
	Content   any   `json:"content"` // string | []ContentBlock
}

func (UserMessage) message()       {}
func (m UserMessage) Role() string { return "user" }

// AssistantMessage is a message from the model.
type AssistantMessage struct {
	Content       []ContentBlock `json:"content"`
	API           Api            `json:"api"`
	Provider      string         `json:"provider"`
	Model         string         `json:"model"`
	ResponseModel string         `json:"responseModel,omitempty"`
	ResponseID    string         `json:"responseId,omitempty"`
	Usage         Usage          `json:"usage"`
	StopReason    StopReason     `json:"stopReason"`
	ErrorMessage  string         `json:"errorMessage,omitempty"`
	Timestamp     int64          `json:"timestamp"`
}

func (AssistantMessage) message()       {}
func (m AssistantMessage) Role() string { return "assistant" }

// ToolResultMessage is the result of a tool call, returned to the model.
type ToolResultMessage struct {
	ToolCallID string         `json:"toolCallId"`
	ToolName   string         `json:"toolName"`
	Content    []ContentBlock `json:"content"`
	Details    any            `json:"details,omitempty"`
	IsError    bool           `json:"isError"`
	Timestamp  int64          `json:"timestamp"`
}

func (ToolResultMessage) message()       {}
func (m ToolResultMessage) Role() string { return "toolResult" }

// Usage reports token usage and cost for an assistant message.
type Usage struct {
	Input       int        `json:"input"`
	Output      int        `json:"output"`
	CacheRead   int        `json:"cacheRead"`
	CacheWrite  int        `json:"cacheWrite"`
	CacheWrite1h int       `json:"cacheWrite1h,omitempty"`
	Reasoning   int        `json:"reasoning,omitempty"`
	TotalTokens int        `json:"totalTokens"`
	Cost        UsageCost  `json:"cost"`
}

// UsageCost is the dollar cost breakdown.
type UsageCost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`
	Total      float64 `json:"total"`
}

// Tool describes a tool the model may invoke. Parameters is a JSON Schema.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// Context is the input to an LLM stream call.
type Context struct {
	SystemPrompt string    `json:"systemPrompt,omitempty"`
	Messages     []Message `json:"messages"`
	Tools        []Tool    `json:"tools,omitempty"`
}

// Model describes a model and which API it uses.
type Model struct {
	ID            string         `json:"id"`
	Name          string         `json:"name"`
	API           Api            `json:"api"`
	Provider      string         `json:"provider"`
	BaseURL       string         `json:"baseUrl"`
	Reasoning     bool           `json:"reasoning"`
	Input         []string       `json:"input"` // e.g. ["text","image"]
	Cost          ModelCost      `json:"cost"`
	ContextWindow int            `json:"contextWindow"`
	MaxTokens     int            `json:"maxTokens"`
	Headers       map[string]string `json:"headers,omitempty"`
}

// ModelCost is the per-million-token dollar cost.
type ModelCost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`
}

// Now returns the current Unix timestamp in milliseconds. (A helper so callers
// do not reach for time directly; also makes timestamps mockable in tests.)
func Now() int64 { return time.Now().UnixMilli() }
