// Package provider holds LLM provider adapters that translate vendor SDK
// streaming APIs into the pi AssistantMessageEvent protocol.
package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// Anthropic is a Provider backed by the official anthropic-sdk-go, using the
// Messages API with streaming and tool use.
type Anthropic struct {
	client    anthropic.Client
	models    []ai.Model
	modelByID map[string]ai.Model
}

// NewAnthropic constructs an Anthropic provider. apiKey falls back to the
// ANTHROPIC_API_KEY env var when empty (the SDK reads it itself, but we pass it
// explicitly so callers can override per-request). Built-in models are
// registered for convenience.
func NewAnthropic(apiKey string, opts ...option.RequestOption) *Anthropic {
	clientOpts := []option.RequestOption{}
	if apiKey != "" {
		clientOpts = append(clientOpts, option.WithAPIKey(apiKey))
	}
	clientOpts = append(clientOpts, opts...)
	client := anthropic.NewClient(clientOpts...)
	a := &Anthropic{
		client:    client,
		modelByID: map[string]ai.Model{},
	}
	for _, m := range builtinAnthropicModels() {
		a.models = append(a.models, m)
		a.modelByID[m.ID] = m
	}
	return a
}

// NewAnthropicFromEnv constructs an Anthropic provider from environment
// variables. It prefers ANTHROPIC_API_KEY, then ANTHROPIC_AUTH_TOKEN (used by
// some Anthropic-compatible gateways), and honors ANTHROPIC_BASE_URL when set.
func NewAnthropicFromEnv() (*Anthropic, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_AUTH_TOKEN")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) is not set")
	}
	var opts []option.RequestOption
	if baseURL := os.Getenv("ANTHROPIC_BASE_URL"); baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	return NewAnthropic(apiKey, opts...), nil
}

// ID returns the provider id.
func (a *Anthropic) ID() string { return "anthropic" }

// Name returns the human-readable provider name.
func (a *Anthropic) Name() string { return "Anthropic" }

// BaseURL returns the configured base URL (empty means the SDK default).
func (a *Anthropic) BaseURL() string { return "" }

// Models returns the known models for this provider.
func (a *Anthropic) Models() []ai.Model { return a.models }

// GetModel returns a registered model by id.
func (a *Anthropic) GetModel(id string) (ai.Model, bool) {
	m, ok := a.modelByID[id]
	return m, ok
}

// RegisterModel adds a custom model to this provider.
func (a *Anthropic) RegisterModel(m ai.Model) {
	a.models = append(a.models, m)
	a.modelByID[m.ID] = m
}

// StreamSimple streams an assistant response from the Anthropic Messages API,
// translating SDK events into the pi AssistantMessageEvent protocol. It honors
// the StreamFn contract: failures are encoded as an ErrorEvent, never returned.
func (a *Anthropic) StreamSimple(ctx context.Context, model ai.Model, context ai.Context, options *ai.SimpleStreamOptions) ai.AssistantMessageEventStream {
	stream := ai.NewAssistantMessageEventStream()
	go func() {
		defer stream.End(nil)
		a.runStream(ctx, model, context, options, stream)
	}()
	return stream
}

func (a *Anthropic) runStream(ctx context.Context, model ai.Model, callCtx ai.Context, options *ai.SimpleStreamOptions, stream ai.AssistantMessageEventStream) {
	runStreamCommon(ctx, model, callCtx, options, stream,
		func(m ai.Model, c ai.Context, o *ai.SimpleStreamOptions) (any, error) {
			return buildAnthropicParams(m, c, o)
		},
		func(m ai.Model) translator { return newAnthropicTranslator(m) },
		func(ctx context.Context, params any) streamSource {
			s := a.client.Messages.NewStreaming(ctx, params.(anthropic.MessageNewParams))
			return newSSEAdapter(s.Next, s.Current, s.Err)
		},
		failureMessage,
	)
}

// anthropicTranslator accumulates streaming events into a partial
// AssistantMessage and emits pi events.
type anthropicTranslator struct {
	model           ai.Model
	partial         ai.AssistantMessage
	blocks          []ai.ContentBlock // accumulated content blocks (parallel to partial.Content)
	blockStarted    map[int]bool
	deltaSeen       map[int]bool
	terminalEmittedField bool
}

func newAnthropicTranslator(model ai.Model) *anthropicTranslator {
	return &anthropicTranslator{
		model:        model,
		blockStarted: map[int]bool{},
		deltaSeen:    map[int]bool{},
		partial: ai.AssistantMessage{
			API:       ai.APIAnthropicMessages,
			Provider:  "anthropic",
			Model:     model.ID,
			Timestamp: ai.Now(),
		},
	}
}

// finalizeToolBlocks decodes accumulated argument JSON for any tool-call
// blocks that received input_json_delta but never saw a content_block_stop
// (some Anthropic-compatible gateways omit stop events). It mutates t.blocks
// in place so the next snapshot reflects decoded arguments.
func (t *anthropicTranslator) finalizeToolBlocks(stream ai.AssistantMessageEventStream) {
	for idx, b := range t.blocks {
		tc, ok := b.(ai.ToolCall)
		if !ok {
			continue
		}
		if len(tc.ArgumentsRaw) > 0 && tc.Arguments == nil {
			var args map[string]any
			if json.Unmarshal(tc.ArgumentsRaw, &args) == nil {
				tc.Arguments = args
			}
		}
		t.blocks[idx] = tc
	}
	t.partial.Content = t.blocks
}

// snapshot returns a copy of the current partial with its content set from the
// accumulated blocks.
func (t *anthropicTranslator) snapshot() ai.AssistantMessage {
	return snapshotBase(t.partial, t.blocks)
}

// finalize satisfies the translator interface; anthropic decodes any tool-call
// args that were streamed but never finalized (gateways omitting stop events).
func (t *anthropicTranslator) finalize(stream ai.AssistantMessageEventStream) {
	t.finalizeToolBlocks(stream)
}

func (t *anthropicTranslator) terminalEmitted() bool { return t.terminalEmittedField }

func (t *anthropicTranslator) handle(event any, stream ai.AssistantMessageEventStream) {
	ev, ok := event.(anthropic.MessageStreamEventUnion)
	if !ok {
		return
	}
	if t.terminalEmittedField {
		return
	}
	switch ev.Type {
	case "message_start":
		// Seed usage from the initial message.
		if ev.Message.ID != "" {
			t.partial.ResponseID = ev.Message.ID
		}
		if ev.Message.Usage.InputTokens > 0 || ev.Message.Usage.OutputTokens > 0 {
			t.partial.Usage = usageFromAnthropic(ev.Message.Usage)
		}
		t.partial.StopReason = mapStopReason(string(ev.Message.StopReason))
		stream.Push(ai.StartEvent{Partial: t.snapshot()})

	case "content_block_start":
		idx := int(ev.Index)
		cb := ev.ContentBlock
		switch cb.Type {
		case "text":
			t.ensureBlock(idx, ai.TextContent{Type: "text"})
			stream.Push(ai.TextStartEvent{ContentIndex: idx, Partial: t.snapshot()})
		case "thinking":
			t.ensureBlock(idx, ai.ThinkingContent{Type: "thinking"})
			stream.Push(ai.ThinkingStartEvent{ContentIndex: idx, Partial: t.snapshot()})
		case "redacted_thinking":
			t.ensureBlock(idx, ai.ThinkingContent{Type: "thinking", Redacted: true, Thinking: cb.Data})
			stream.Push(ai.ThinkingStartEvent{ContentIndex: idx, Partial: t.snapshot()})
		case "tool_use":
			tc := ai.ToolCall{Type: "toolCall", ID: cb.ID, Name: cb.Name}
			// Some gateways send the full input object on content_block_start
			// instead of streaming it via input_json_delta. Capture it as a
			// fallback ONLY when it carries real keys; the standard protocol
			// sends an empty {} placeholder here and the real args via deltas.
			if m, ok := cb.Input.(map[string]any); ok && len(m) > 0 {
				if raw, err := json.Marshal(m); err == nil {
					tc.ArgumentsRaw = raw
					tc.Arguments = m
				}
			}
			t.ensureBlock(idx, tc)
			stream.Push(ai.ToolCallStartEvent{ContentIndex: idx, Partial: t.snapshot()})
		}

	case "content_block_delta":
		idx := int(ev.Index)
		d := ev.Delta
		switch d.Type {
		case "text_delta":
			if tc, ok := t.blocks[idx].(ai.TextContent); ok {
				tc.Text += d.Text
				t.blocks[idx] = tc
				t.partial.Content = t.blocks
				stream.Push(ai.TextDeltaEvent{ContentIndex: idx, Delta: d.Text, Partial: t.snapshot()})
			}
		case "thinking_delta":
			if tc, ok := t.blocks[idx].(ai.ThinkingContent); ok {
				tc.Thinking += d.Thinking
				t.blocks[idx] = tc
				t.partial.Content = t.blocks
				stream.Push(ai.ThinkingDeltaEvent{ContentIndex: idx, Delta: d.Thinking, Partial: t.snapshot()})
			}
		case "signature_delta":
			if tc, ok := t.blocks[idx].(ai.ThinkingContent); ok {
				tc.ThinkingSignature += d.Signature
				t.blocks[idx] = tc
				t.partial.Content = t.blocks
			}
		case "input_json_delta":
			if tc, ok := t.blocks[idx].(ai.ToolCall); ok {
				// Some gateways send the full input object on content_block_start
				// (captured into ArgumentsRaw/Arguments when len>0) instead of
				// streaming it via deltas. If start already gave real, complete args,
				// treat start as authoritative and ignore subsequent deltas: sending
				// both is malformed, and reassembling from fragments would discard
				// start's args (the bug). Standard protocol sends an empty {} here.
				if len(tc.ArgumentsRaw) > 0 && len(tc.Arguments) > 0 && !t.deltaSeen[idx] {
					t.deltaSeen[idx] = true
				} else {
					t.deltaSeen[idx] = true
					tc.ArgumentsRaw = append(tc.ArgumentsRaw, []byte(d.PartialJSON)...)
					t.blocks[idx] = tc
					t.partial.Content = t.blocks
					stream.Push(ai.ToolCallDeltaEvent{ContentIndex: idx, Delta: d.PartialJSON, Partial: t.snapshot()})
				}
			}
		}

	case "content_block_stop":
		idx := int(ev.Index)
		switch b := t.blocks[idx].(type) {
		case ai.TextContent:
			stream.Push(ai.TextEndEvent{ContentIndex: idx, Content: b.Text, Partial: t.snapshot()})
		case ai.ThinkingContent:
			stream.Push(ai.ThinkingEndEvent{ContentIndex: idx, Content: b.Thinking, Partial: t.snapshot()})
		case ai.ToolCall:
			// Decode accumulated argument JSON; on failure leave Arguments empty.
			if len(b.ArgumentsRaw) > 0 {
				var args map[string]any
				if json.Unmarshal(b.ArgumentsRaw, &args) == nil {
					b.Arguments = args
				}
			}
			t.blocks[idx] = b
			stream.Push(ai.ToolCallEndEvent{ContentIndex: idx, ToolCall: b, Partial: t.snapshot()})
		}

	case "message_delta":
		if ev.Delta.StopReason != "" {
			t.partial.StopReason = mapStopReason(string(ev.Delta.StopReason))
		}
		t.partial.Usage = usageFromDelta(t.partial.Usage, ev.Usage)

	case "message_stop":
		msg := t.snapshot()
		if msg.StopReason == "" {
			msg.StopReason = ai.StopStop
		}
		t.terminalEmittedField = true
		stream.Push(ai.DoneEvent{Reason: msg.StopReason, Message: msg})
	}
}

// ensureBlock makes sure a content block slot at idx exists, initialising it to
// init when absent, and records that the partial has started.
func (t *anthropicTranslator) ensureBlock(idx int, init ai.ContentBlock) {
	for len(t.blocks) <= idx {
		t.blocks = append(t.blocks, nil)
	}
	if !t.blockStarted[idx] {
		t.blocks[idx] = init
		t.blockStarted[idx] = true
		t.partial.Content = t.blocks
	}
}

func usageFromAnthropic(u anthropic.Usage) ai.Usage {
	return ai.Usage{
		Input:       int(u.InputTokens),
		Output:      int(u.OutputTokens),
		CacheRead:   int(u.CacheReadInputTokens),
		CacheWrite:  int(u.CacheCreationInputTokens),
		TotalTokens: int(u.InputTokens + u.OutputTokens),
	}
}

func usageFromDelta(base ai.Usage, u anthropic.MessageDeltaUsage) ai.Usage {
	out := base
	if u.InputTokens > 0 {
		out.Input = int(u.InputTokens)
	}
	if u.OutputTokens > 0 {
		out.Output = int(u.OutputTokens)
	}
	if u.CacheReadInputTokens > 0 {
		out.CacheRead = int(u.CacheReadInputTokens)
	}
	if u.CacheCreationInputTokens > 0 {
		out.CacheWrite = int(u.CacheCreationInputTokens)
	}
	out.TotalTokens = out.Input + out.Output
	return out
}

func mapStopReason(s string) ai.StopReason {
	switch s {
	case "end_turn", "stop_sequence":
		return ai.StopStop
	case "max_tokens":
		return ai.StopLength
	case "tool_use":
		return ai.StopToolUse
	case "":
		return ""
	}
	return ai.StopStop
}

func failureMessage(model ai.Model, msg string, aborted bool) ai.AssistantMessage {
	return makeFailureMessage(model, ai.APIAnthropicMessages, "anthropic", msg, aborted)
}

// buildAnthropicParams converts a pi Context and options into an Anthropic
// MessageNewParams.
func buildAnthropicParams(model ai.Model, context ai.Context, options *ai.SimpleStreamOptions) (anthropic.MessageNewParams, error) {
	if options == nil {
		options = &ai.SimpleStreamOptions{}
	}

	maxTokens := int64(options.MaxTokens)
	if maxTokens == 0 {
		maxTokens = int64(model.MaxTokens)
	}
	if maxTokens == 0 {
		maxTokens = 4096
	}

	params := anthropic.MessageNewParams{
		MaxTokens: maxTokens,
		Model:     anthropic.Model(model.ID),
	}

	if context.SystemPrompt != "" {
		params.System = []anthropic.TextBlockParam{{
			Text: context.SystemPrompt,
		}}
	}

	msgs, err := convertMessages(context.Messages)
	if err != nil {
		return anthropic.MessageNewParams{}, err
	}
	params.Messages = msgs

	if len(context.Tools) > 0 {
		params.Tools = convertTools(context.Tools)
	}

	if options.Temperature != nil {
		params.Temperature = anthropic.Float(*options.Temperature)
	}

	return params, nil
}

func convertMessages(messages []ai.Message) ([]anthropic.MessageParam, error) {
	var out []anthropic.MessageParam
	for _, m := range messages {
		switch msg := m.(type) {
		case ai.UserMessage:
			blocks, err := convertUserContent(msg.Content)
			if err != nil {
				return nil, err
			}
			out = append(out, anthropic.NewUserMessage(blocks...))
		case ai.AssistantMessage:
			blocks := convertAssistantContent(msg.Content)
			if len(blocks) > 0 {
				out = append(out, anthropic.NewAssistantMessage(blocks...))
			}
		case ai.ToolResultMessage:
			out = append(out, anthropic.MessageParam{
				Role:    "user",
				Content: []anthropic.ContentBlockParamUnion{convertToolResult(msg)},
			})
		}
	}
	return out, nil
}

func convertUserContent(content any) ([]anthropic.ContentBlockParamUnion, error) {
	switch c := content.(type) {
	case string:
		return []anthropic.ContentBlockParamUnion{anthropic.NewTextBlock(c)}, nil
	case []ai.ContentBlock:
		var out []anthropic.ContentBlockParamUnion
		for _, b := range c {
			switch block := b.(type) {
			case ai.TextContent:
				out = append(out, anthropic.NewTextBlock(block.Text))
			case ai.ImageContent:
				out = append(out, anthropic.NewImageBlock(anthropic.Base64ImageSourceParam{
					Data:      block.Data,
					MediaType: anthropic.Base64ImageSourceMediaType(block.MimeType),
				}))
			}
		}
		return out, nil
	}
	return nil, fmt.Errorf("unsupported user content type: %T", content)
}

func convertAssistantContent(blocks []ai.ContentBlock) []anthropic.ContentBlockParamUnion {
	var out []anthropic.ContentBlockParamUnion
	for _, b := range blocks {
		switch block := b.(type) {
		case ai.TextContent:
			if block.Text != "" {
				out = append(out, anthropic.NewTextBlock(block.Text))
			}
		case ai.ToolCall:
			out = append(out, anthropic.NewToolUseBlock(block.ID, decodeArgs(block), block.Name))
		}
	}
	return out
}

func convertToolResult(msg ai.ToolResultMessage) anthropic.ContentBlockParamUnion {
	var text string
	for _, b := range msg.Content {
		if tc, ok := b.(ai.TextContent); ok {
			if text != "" {
				text += "\n"
			}
			text += tc.Text
		}
	}
	return anthropic.NewToolResultBlock(msg.ToolCallID, text, msg.IsError)
}

func convertTools(tools []ai.Tool) []anthropic.ToolUnionParam {
	out := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		schema := anthropic.ToolInputSchemaParam{
			Properties:   t.Parameters["properties"],
			Required:     requiredFromSchema(t.Parameters),
			ExtraFields:  extraSchemaFields(t.Parameters),
		}
		out = append(out, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        t.Name,
				Description: anthropic.String(t.Description),
				InputSchema: schema,
			},
		})
	}
	return out
}

func requiredFromSchema(params map[string]any) []string {
	if r, ok := params["required"].([]any); ok {
		out := make([]string, 0, len(r))
		for _, x := range r {
			if s, ok := x.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

// extraSchemaFields returns schema keys other than type/properties/required so
// they survive into the Anthropic input schema.
func extraSchemaFields(params map[string]any) map[string]any {
	extra := map[string]any{}
	for k, v := range params {
		switch k {
		case "type", "properties", "required":
			continue
		}
		extra[k] = v
	}
	if len(extra) == 0 {
		return nil
	}
	return extra
}

func decodeArgs(tc ai.ToolCall) any {
	if len(tc.ArgumentsRaw) > 0 {
		var v any
		if json.Unmarshal(tc.ArgumentsRaw, &v) == nil {
			return v
		}
	}
	return tc.Arguments
}
