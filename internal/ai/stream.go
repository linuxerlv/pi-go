package ai

import "context"

// ThinkingLevel is the reasoning/thinking effort requested for a turn.
type ThinkingLevel string

const (
	ThinkingOff     ThinkingLevel = "off"
	ThinkingMinimal ThinkingLevel = "minimal"
	ThinkingLow     ThinkingLevel = "low"
	ThinkingMedium  ThinkingLevel = "medium"
	ThinkingHigh    ThinkingLevel = "high"
	ThinkingXHigh   ThinkingLevel = "xhigh"
)

// StreamOptions are the base options shared by all provider stream calls.
// Provider-specific options live on the concrete provider adapter.
type StreamOptions struct {
	Temperature *float64
	MaxTokens   int
	APIKey      string
	TimeoutMs   int
	MaxRetries  int
	Headers     map[string]string
	Metadata    map[string]any
	// SessionID is an optional session id for providers that support
	// session-based caching/routing.
	SessionID string
}

// SimpleStreamOptions extends StreamOptions with reasoning controls. This is
// the option type accepted by streamSimple, the entry point the agent loop
// calls.
type SimpleStreamOptions struct {
	StreamOptions
	Reasoning       ThinkingLevel
	ThinkingBudgets ThinkingBudgets
}

// ThinkingBudgets maps thinking levels to token budgets (token-based providers).
type ThinkingBudgets struct {
	Minimal int
	Low     int
	Medium  int
	High    int
}

// StreamFn is the stream function signature the agent loop depends on.
//
// Contract (mirrors @earendil-works/pi-ai StreamFunction):
//   - Must return an AssistantMessageEventStream; it must NOT return an error
//     for request/model/runtime failures. Failures are encoded in the returned
//     stream as an ErrorEvent carrying an AssistantMessage with stopReason
//     "error" or "aborted" and a populated ErrorMessage.
//   - The ctx is used for cancellation; an aborted request yields an
//     ErrorEvent with stopReason "aborted".
type StreamFn func(ctx context.Context, model Model, context Context, options *SimpleStreamOptions) AssistantMessageEventStream

// Provider is the runtime unit owning a set of models and stream behavior. It
// mirrors @earendil-works/pi-ai's Provider interface.
type Provider interface {
	ID() string
	Name() string
	BaseURL() string
	Models() []Model
	// GetModel returns a registered model by id.
	GetModel(id string) (Model, bool)
	// StreamSimple streams an assistant response, translating the provider's
	// native events into the pi AssistantMessageEvent protocol. It must honor
	// the StreamFn contract (no error return; failures become ErrorEvent).
	StreamSimple(ctx context.Context, model Model, context Context, options *SimpleStreamOptions) AssistantMessageEventStream
}

// Models is a collection of providers plus convenience stream dispatch. It
// mirrors @earendil-works/pi-ai's Models interface.
type Models interface {
	Providers() []Provider
	GetProvider(id string) Provider
	GetModel(provider, id string) (Model, bool)
	// StreamSimple dispatches to the provider that owns the model.
	StreamSimple(ctx context.Context, model Model, context Context, options *SimpleStreamOptions) AssistantMessageEventStream
}
