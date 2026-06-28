package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// OpenAI is a Provider backed by the official openai-go SDK, using the Chat
// Completions API with streaming and tool use. It supports any OpenAI-compatible
// endpoint (OpenRouter, local vLLM, Ollama, ...) via a custom base URL.
type OpenAI struct {
	client    openai.Client
	models    []ai.Model
	modelByID map[string]ai.Model
	baseURL   string
}

// NewOpenAI constructs an OpenAI provider. apiKey is the bearer key; baseURL
// overrides the API endpoint ("" means the SDK default, https://api.openai.com/v1).
func NewOpenAI(apiKey, baseURL string, opts ...option.RequestOption) *OpenAI {
	clientOpts := []option.RequestOption{}
	if apiKey != "" {
		clientOpts = append(clientOpts, option.WithAPIKey(apiKey))
	}
	if baseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(baseURL))
	}
	clientOpts = append(clientOpts, opts...)
	client := openai.NewClient(clientOpts...)
	p := &OpenAI{
		client:    client,
		modelByID: map[string]ai.Model{},
		baseURL:   baseURL,
	}
	registerBuiltinOpenAIModels(p)
	return p
}

// NewOpenAIFromEnv constructs an OpenAI provider from environment variables:
// OPENAI_API_KEY (or OPENAI_AUTH_TOKEN) and OPENAI_BASE_URL.
func NewOpenAIFromEnv() (*OpenAI, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_AUTH_TOKEN")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY (or OPENAI_AUTH_TOKEN) is not set")
	}
	return NewOpenAI(apiKey, os.Getenv("OPENAI_BASE_URL")), nil
}

// ID returns the provider id.
func (p *OpenAI) ID() string { return "openai" }

// Name returns the human-readable provider name.
func (p *OpenAI) Name() string {
	if p.baseURL != "" {
		return "OpenAI-compatible"
	}
	return "OpenAI"
}

// BaseURL returns the configured base URL.
func (p *OpenAI) BaseURL() string { return p.baseURL }

// Models returns the known models for this provider.
func (p *OpenAI) Models() []ai.Model { return p.models }

// GetModel returns a registered model by id.
func (p *OpenAI) GetModel(id string) (ai.Model, bool) {
	m, ok := p.modelByID[id]
	return m, ok
}

// RegisterModel adds a model to this provider.
func (p *OpenAI) RegisterModel(m ai.Model) {
	p.models = append(p.models, m)
	p.modelByID[m.ID] = m
}

// StreamSimple streams an assistant response from the OpenAI Chat Completions
// API, translating SDK chunks into the pi AssistantMessageEvent protocol.
func (p *OpenAI) StreamSimple(ctx context.Context, model ai.Model, context ai.Context, options *ai.SimpleStreamOptions) ai.AssistantMessageEventStream {
	stream := ai.NewAssistantMessageEventStream()
	go func() {
		defer stream.End(nil)
		p.runStream(ctx, model, context, options, stream)
	}()
	return stream
}

func (p *OpenAI) runStream(ctx context.Context, model ai.Model, callCtx ai.Context, options *ai.SimpleStreamOptions, stream ai.AssistantMessageEventStream) {
	runStreamCommon(ctx, model, callCtx, options, stream,
		func(m ai.Model, c ai.Context, o *ai.SimpleStreamOptions) (any, error) {
			return buildOpenAIParams(m, c, o)
		},
		func(m ai.Model) translator { return newOpenAITranslator(m) },
		func(ctx context.Context, params any) streamSource {
			s := p.client.Chat.Completions.NewStreaming(ctx, params.(openai.ChatCompletionNewParams))
			return newSSEAdapter(s.Next, s.Current, s.Err)
		},
		openAIFailureMessage,
	)
}

// openAITranslator accumulates Chat Completion chunks into a partial
// AssistantMessage and emits pi events. Tool calls are accumulated by their
// OpenAI-assigned index across deltas.
type openAITranslator struct {
	model           ai.Model
	partial         ai.AssistantMessage
	blocks          []ai.ContentBlock
	started         bool
	textStarted     bool
	toolCalls       map[int]*ai.ToolCall // by OpenAI tool-call index
	toolOrder       []int                // indices in arrival order
	terminalEmittedField bool
}

func newOpenAITranslator(model ai.Model) *openAITranslator {
	return &openAITranslator{
		model:     model,
		toolCalls: map[int]*ai.ToolCall{},
		partial: ai.AssistantMessage{
			API:       ai.APIOpenAICompletions,
			Provider:  "openai",
			Model:     model.ID,
			Timestamp: ai.Now(),
		},
	}
}

func (t *openAITranslator) snapshot() ai.AssistantMessage {
	return snapshotBase(t.partial, t.blocks)
}

func (t *openAITranslator) terminalEmitted() bool { return t.terminalEmittedField }

func (t *openAITranslator) handle(event any, stream ai.AssistantMessageEventStream) {
	chunk, ok := event.(openai.ChatCompletionChunk)
	if !ok {
		return
	}
	if t.terminalEmittedField {
		return
	}
	if !t.started {
		t.started = true
		stream.Push(ai.StartEvent{Partial: t.snapshot()})
	}
	// Usage (when stream_options.include_usage is set, arrives on the final
	// chunk with empty choices).
	if chunk.Usage.TotalTokens > 0 {
		t.partial.Usage = usageFromOpenAI(chunk.Usage)
	}

	for _, choice := range chunk.Choices {
		if choice.FinishReason != "" {
			t.partial.StopReason = mapOpenAIStopReason(choice.FinishReason)
		}
		delta := choice.Delta

		// Text content delta.
		if delta.Content != "" {
			if !t.textStarted {
				t.blocks = append(t.blocks, ai.TextContent{Type: "text"})
				t.textStarted = true
				idx := len(t.blocks) - 1
				stream.Push(ai.TextStartEvent{ContentIndex: idx, Partial: t.snapshot()})
			}
			idx := len(t.blocks) - 1
			if tc, ok := t.blocks[idx].(ai.TextContent); ok {
				tc.Text += delta.Content
				t.blocks[idx] = tc
				t.partial.Content = t.blocks
				stream.Push(ai.TextDeltaEvent{ContentIndex: idx, Delta: delta.Content, Partial: t.snapshot()})
			}
		}

		// Tool call deltas (accumulated by index).
		for _, tcd := range delta.ToolCalls {
			idx := int(tcd.Index)
			tc, exists := t.toolCalls[idx]
			if !exists {
				tc = &ai.ToolCall{Type: "toolCall"}
				t.toolCalls[idx] = tc
				t.toolOrder = append(t.toolOrder, idx)
				// Append a new tool-call block; its final index in t.blocks is
				// len(t.blocks) at the time of creation.
				t.blocks = append(t.blocks, *tc)
				stream.Push(ai.ToolCallStartEvent{ContentIndex: len(t.blocks) - 1, Partial: t.snapshot()})
			}
			if tcd.ID != "" {
				tc.ID = tcd.ID
			}
			if tcd.Function.Name != "" {
				tc.Name = tcd.Function.Name
			}
			if tcd.Function.Arguments != "" {
				tc.ArgumentsRaw = append(tc.ArgumentsRaw, []byte(tcd.Function.Arguments)...)
				stream.Push(ai.ToolCallDeltaEvent{
					ContentIndex: t.blockIndexForTool(idx),
					Delta:        tcd.Function.Arguments,
					Partial:      t.snapshot(),
				})
			}
			// Write back the accumulated tool call into its block slot.
			if bi := t.blockIndexForTool(idx); bi >= 0 {
				t.blocks[bi] = *tc
				t.partial.Content = t.blocks
			}
		}
	}
}

// blockIndexForTool maps an OpenAI tool-call index to its position in t.blocks.
// Tool-call blocks are appended after any text block, in toolOrder.
func (t *openAITranslator) blockIndexForTool(toolIdx int) int {
	textOffset := 0
	if t.textStarted {
		textOffset = 1
	}
	for i, ti := range t.toolOrder {
		if ti == toolIdx {
			return textOffset + i
		}
	}
	return -1
}

// finalize decodes accumulated tool-call argument JSON and emits toolcall_end
// (and text_end if pending) before the terminal event.
func (t *openAITranslator) finalize(stream ai.AssistantMessageEventStream) {
	// Close text block.
	if t.textStarted {
		idx := 0
		if tc, ok := t.blocks[idx].(ai.TextContent); ok {
			stream.Push(ai.TextEndEvent{ContentIndex: idx, Content: tc.Text, Partial: t.snapshot()})
		}
	}
	// Decode tool-call arguments and emit toolcall_end.
	for _, ti := range t.toolOrder {
		tc := t.toolCalls[ti]
		if len(tc.ArgumentsRaw) > 0 {
			var args map[string]any
			if json.Unmarshal(tc.ArgumentsRaw, &args) == nil {
				tc.Arguments = args
			}
		}
		bi := t.blockIndexForTool(ti)
		if bi >= 0 {
			t.blocks[bi] = *tc
		}
		stream.Push(ai.ToolCallEndEvent{ContentIndex: bi, ToolCall: *tc, Partial: t.snapshot()})
	}
	t.partial.Content = t.blocks
}

func usageFromOpenAI(u openai.CompletionUsage) ai.Usage {
	out := ai.Usage{
		Input:       int(u.PromptTokens),
		Output:      int(u.CompletionTokens),
		TotalTokens: int(u.TotalTokens),
	}
	if u.PromptTokensDetails.CachedTokens > 0 {
		out.CacheRead = int(u.PromptTokensDetails.CachedTokens)
	}
	return out
}

func mapOpenAIStopReason(s string) ai.StopReason {
	switch s {
	case "stop":
		return ai.StopStop
	case "length":
		return ai.StopLength
	case "tool_calls", "function_call":
		return ai.StopToolUse
	case "content_filter":
		return ai.StopStop
	case "":
		return ""
	}
	return ai.StopStop
}

func openAIFailureMessage(model ai.Model, msg string, aborted bool) ai.AssistantMessage {
	reason := ai.StopError
	if aborted {
		reason = ai.StopAborted
	}
	return ai.AssistantMessage{
		Content:      []ai.ContentBlock{ai.TextContent{Type: "text", Text: ""}},
		API:          ai.APIOpenAICompletions,
		Provider:     "openai",
		Model:        model.ID,
		StopReason:   reason,
		ErrorMessage: msg,
		Timestamp:    ai.Now(),
	}
}

// buildOpenAIParams converts a pi Context and options into ChatCompletionNewParams.
func buildOpenAIParams(model ai.Model, context ai.Context, options *ai.SimpleStreamOptions) (openai.ChatCompletionNewParams, error) {
	if options == nil {
		options = &ai.SimpleStreamOptions{}
	}

	params := openai.ChatCompletionNewParams{
		Model: openai.ChatModel(model.ID),
	}

	if context.SystemPrompt != "" {
		params.Messages = append(params.Messages, openai.ChatCompletionMessageParamUnion{
			OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: openai.String(context.SystemPrompt),
				},
			},
		})
	}

	msgs, err := convertOpenAIMessages(context.Messages)
	if err != nil {
		return openai.ChatCompletionNewParams{}, err
	}
	params.Messages = append(params.Messages, msgs...)

	if len(context.Tools) > 0 {
		params.Tools = convertOpenAITools(context.Tools)
	}

	if options.MaxTokens > 0 {
		params.MaxTokens = openai.Int(int64(options.MaxTokens))
	}
	if options.Temperature != nil {
		params.Temperature = openai.Float(*options.Temperature)
	}
	return params, nil
}

func convertOpenAIMessages(messages []ai.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	var out []openai.ChatCompletionMessageParamUnion
	for _, m := range messages {
		switch msg := m.(type) {
		case ai.UserMessage:
			out = append(out, openai.ChatCompletionMessageParamUnion{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String(userText(msg.Content)),
					},
				},
			})
		case ai.AssistantMessage:
			amp := &openai.ChatCompletionAssistantMessageParam{}
			var textParts []string
			for _, b := range msg.Content {
				switch block := b.(type) {
				case ai.TextContent:
					textParts = append(textParts, block.Text)
				case ai.ToolCall:
					amp.ToolCalls = append(amp.ToolCalls, openai.ChatCompletionMessageToolCallUnionParam{
						OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
							ID: block.ID,
							Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
								Arguments: string(block.ArgumentsRaw),
								Name:      block.Name,
							},
						},
					})
				}
			}
			if len(textParts) > 0 {
				amp.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(strings.Join(textParts, "")),
				}
			}
			out = append(out, openai.ChatCompletionMessageParamUnion{OfAssistant: amp})
		case ai.ToolResultMessage:
			out = append(out, openai.ChatCompletionMessageParamUnion{
				OfTool: &openai.ChatCompletionToolMessageParam{
					ToolCallID: msg.ToolCallID,
					Content: openai.ChatCompletionToolMessageParamContentUnion{
						OfString: openai.String(toolResultText(msg)),
					},
				},
			})
		}
	}
	return out, nil
}

func convertOpenAITools(tools []ai.Tool) []openai.ChatCompletionToolUnionParam {
	out := make([]openai.ChatCompletionToolUnionParam, 0, len(tools))
	for _, t := range tools {
		out = append(out, openai.ChatCompletionToolUnionParam{
			OfFunction: &openai.ChatCompletionFunctionToolParam{
				Function: shared.FunctionDefinitionParam{
					Name:        t.Name,
					Description: openai.String(t.Description),
					Parameters:  shared.FunctionParameters(t.Parameters),
				},
			},
		})
	}
	return out
}

func userText(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []ai.ContentBlock:
		var parts []string
		for _, b := range c {
			if tc, ok := b.(ai.TextContent); ok {
				parts = append(parts, tc.Text)
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

func toolResultText(msg ai.ToolResultMessage) string {
	var parts []string
	for _, b := range msg.Content {
		if tc, ok := b.(ai.TextContent); ok {
			parts = append(parts, tc.Text)
		}
	}
	return strings.Join(parts, "\n")
}
