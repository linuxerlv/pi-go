package provider

import (
	"context"
	"encoding/json"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// OpenAIResponses is a Provider backed by the OpenAI Responses API
// (responses.NewStreaming). It supports the same OpenAI-compatible endpoints
// as OpenAI (via base URL) but uses the Responses event model. Some providers
// (e.g. OpenAI itself) expose tools like web_search through this API.
type OpenAIResponses struct {
	client    openai.Client
	models    []ai.Model
	modelByID map[string]ai.Model
	baseURL   string
}

// NewOpenAIResponses constructs a Responses-API provider.
func NewOpenAIResponses(apiKey, baseURL string, opts ...option.RequestOption) *OpenAIResponses {
	clientOpts := []option.RequestOption{}
	if apiKey != "" {
		clientOpts = append(clientOpts, option.WithAPIKey(apiKey))
	}
	if baseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(baseURL))
	}
	clientOpts = append(clientOpts, opts...)
	p := &OpenAIResponses{
		client:    openai.NewClient(clientOpts...),
		modelByID: map[string]ai.Model{},
		baseURL:   baseURL,
	}
	for _, m := range builtinOpenAIResponsesModels() {
		p.models = append(p.models, m)
		p.modelByID[m.ID] = m
	}
	return p
}

// ID returns the provider id.
func (p *OpenAIResponses) ID() string { return "openai-responses" }

// Name returns the human-readable name.
func (p *OpenAIResponses) Name() string {
	if p.baseURL != "" {
		return "OpenAI Responses (compatible)"
	}
	return "OpenAI Responses"
}

// BaseURL returns the configured base URL.
func (p *OpenAIResponses) BaseURL() string { return p.baseURL }

// Models returns known models.
func (p *OpenAIResponses) Models() []ai.Model { return p.models }

// GetModel returns a registered model by id.
func (p *OpenAIResponses) GetModel(id string) (ai.Model, bool) {
	m, ok := p.modelByID[id]
	return m, ok
}

// RegisterModel adds a model.
func (p *OpenAIResponses) RegisterModel(m ai.Model) {
	p.models = append(p.models, m)
	p.modelByID[m.ID] = m
}

// StreamSimple streams an assistant response via the Responses API.
func (p *OpenAIResponses) StreamSimple(ctx context.Context, model ai.Model, context ai.Context, options *ai.SimpleStreamOptions) ai.AssistantMessageEventStream {
	stream := ai.NewAssistantMessageEventStream()
	go func() {
		defer stream.End(nil)
		p.runStream(ctx, model, context, options, stream)
	}()
	return stream
}

func (p *OpenAIResponses) runStream(ctx context.Context, model ai.Model, context ai.Context, options *ai.SimpleStreamOptions, stream ai.AssistantMessageEventStream) {
	params, err := buildResponsesParams(model, context, options)
	if err != nil {
		stream.Push(ai.ErrorEvent{Reason: ai.StopError, Error: openAIFailureMessage(model, err.Error(), false)})
		return
	}
	sdkStream := p.client.Responses.NewStreaming(ctx, params)
	tr := newResponsesTranslator(model)
	started := false

	for sdkStream.Next() {
		if ctx.Err() != nil {
			stream.Push(ai.ErrorEvent{Reason: ai.StopAborted, Error: openAIFailureMessage(model, "Operation aborted", true)})
			return
		}
		ev := sdkStream.Current()
		if !started {
			started = true
			stream.Push(ai.StartEvent{Partial: tr.snapshot()})
		}
		tr.handle(ev, stream)
	}
	if err := sdkStream.Err(); err != nil {
		reason := ai.StopError
		aborted := false
		if ctx.Err() != nil {
			reason = ai.StopAborted
			aborted = true
		}
		stream.Push(ai.ErrorEvent{Reason: reason, Error: openAIFailureMessage(model, err.Error(), aborted)})
		return
	}
	tr.finalize(stream)
	if !tr.terminalEmitted {
		msg := tr.snapshot()
		if msg.StopReason == "" {
			msg.StopReason = ai.StopStop
		}
		stream.Push(ai.DoneEvent{Reason: msg.StopReason, Message: msg})
	}
}

// responsesTranslator accumulates Responses stream events into a partial
// AssistantMessage. Text and function-call arguments are accumulated by
// output index / item id.
type responsesTranslator struct {
	model           ai.Model
	partial         ai.AssistantMessage
	blocks          []ai.ContentBlock
	textStarted     bool
	// tool calls indexed by output index (the Responses API assigns each
	// function_call item an output_index).
	toolCalls      map[int]*ai.ToolCall
	toolOrder      []int
	terminalEmitted bool
}

func newResponsesTranslator(model ai.Model) *responsesTranslator {
	return &responsesTranslator{
		model:     model,
		toolCalls: map[int]*ai.ToolCall{},
		partial: ai.AssistantMessage{
			API:       ai.APIOpenAIResponses,
			Provider:  "openai-responses",
			Model:     model.ID,
			Timestamp: ai.Now(),
		},
	}
}

func (t *responsesTranslator) snapshot() ai.AssistantMessage {
	msg := t.partial
	msg.Content = append([]ai.ContentBlock(nil), t.blocks...)
	return msg
}

func (t *responsesTranslator) handle(ev responses.ResponseStreamEventUnion, stream ai.AssistantMessageEventStream) {
	if t.terminalEmitted {
		return
	}
	switch ev.Type {
	case "response.output_text.delta":
		if !t.textStarted {
			t.blocks = append(t.blocks, ai.TextContent{Type: "text"})
			t.textStarted = true
			stream.Push(ai.TextStartEvent{ContentIndex: 0, Partial: t.snapshot()})
		}
		if tc, ok := t.blocks[0].(ai.TextContent); ok {
			tc.Text += ev.Delta
			t.blocks[0] = tc
			t.partial.Content = t.blocks
			stream.Push(ai.TextDeltaEvent{ContentIndex: 0, Delta: ev.Delta, Partial: t.snapshot()})
		}

	case "response.output_item.added":
		// A function_call item starts here, carrying its id and name.
		item := ev.Item
		if item.Type == "function_call" {
			idx := int(ev.OutputIndex)
			tc := &ai.ToolCall{Type: "toolCall", ID: item.CallID, Name: item.Name}
			t.toolCalls[idx] = tc
			t.toolOrder = append(t.toolOrder, idx)
			t.blocks = append(t.blocks, *tc)
			stream.Push(ai.ToolCallStartEvent{ContentIndex: t.blockIndexForTool(idx), Partial: t.snapshot()})
		}

	case "response.function_call_arguments.delta":
		idx := int(ev.OutputIndex)
		tc, ok := t.toolCalls[idx]
		if !ok {
			// Some servers don't send output_item.added for function calls; create on demand.
			tc = &ai.ToolCall{Type: "toolCall", ID: ev.ItemID}
			t.toolCalls[idx] = tc
			t.toolOrder = append(t.toolOrder, idx)
			t.blocks = append(t.blocks, *tc)
			stream.Push(ai.ToolCallStartEvent{ContentIndex: t.blockIndexForTool(idx), Partial: t.snapshot()})
		}
		tc.ArgumentsRaw = append(tc.ArgumentsRaw, []byte(ev.Arguments)...)
		if bi := t.blockIndexForTool(idx); bi >= 0 {
			t.blocks[bi] = *tc
			t.partial.Content = t.blocks
			stream.Push(ai.ToolCallDeltaEvent{ContentIndex: bi, Delta: ev.Arguments, Partial: t.snapshot()})
		}

	case "response.function_call_arguments.done":
		idx := int(ev.OutputIndex)
		if tc, ok := t.toolCalls[idx]; ok {
			if len(tc.ArgumentsRaw) == 0 && ev.Arguments != "" {
				tc.ArgumentsRaw = []byte(ev.Arguments)
			}
			if bi := t.blockIndexForTool(idx); bi >= 0 {
				t.blocks[bi] = *tc
			}
		}

	case "response.completed":
		resp := ev.Response
		t.partial.Usage = usageFromResponses(resp.Usage)
		t.partial.StopReason = mapResponsesStatus(string(resp.Status))
		// Status may indicate tool use if any function calls were emitted.
		if len(t.toolOrder) > 0 && t.partial.StopReason == "" {
			t.partial.StopReason = ai.StopToolUse
		}
	case "response.failed", "response.incomplete", "error":
		msg := ev.Message
		if msg == "" {
			msg = ev.Code
		}
		if msg == "" {
			msg = "responses stream error: " + ev.Type
		}
		m := t.snapshot()
		m.StopReason = ai.StopError
		m.ErrorMessage = msg
		t.terminalEmitted = true
		stream.Push(ai.ErrorEvent{Reason: ai.StopError, Error: m})
	}
}

func (t *responsesTranslator) finalize(stream ai.AssistantMessageEventStream) {
	if t.textStarted {
		if tc, ok := t.blocks[0].(ai.TextContent); ok {
			stream.Push(ai.TextEndEvent{ContentIndex: 0, Content: tc.Text, Partial: t.snapshot()})
		}
	}
	for _, idx := range t.toolOrder {
		tc := t.toolCalls[idx]
		if len(tc.ArgumentsRaw) > 0 {
			var args map[string]any
			if json.Unmarshal(tc.ArgumentsRaw, &args) == nil {
				tc.Arguments = args
			}
		}
		bi := t.blockIndexForTool(idx)
		if bi >= 0 {
			t.blocks[bi] = *tc
		}
		stream.Push(ai.ToolCallEndEvent{ContentIndex: bi, ToolCall: *tc, Partial: t.snapshot()})
	}
	t.partial.Content = t.blocks
}

func (t *responsesTranslator) blockIndexForTool(toolIdx int) int {
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

func usageFromResponses(u responses.ResponseUsage) ai.Usage {
	out := ai.Usage{
		Input:       int(u.InputTokens),
		Output:      int(u.OutputTokens),
		TotalTokens: int(u.TotalTokens),
	}
	if u.InputTokensDetails.CachedTokens > 0 {
		out.CacheRead = int(u.InputTokensDetails.CachedTokens)
	}
	return out
}

func mapResponsesStatus(status string) ai.StopReason {
	switch status {
	case "completed":
		return ai.StopStop
	case "incomplete":
		return ai.StopLength
	case "failed", "cancelled":
		return ai.StopError
	}
	return ""
}

// buildResponsesParams converts a pi Context into ResponseNewParams.
func buildResponsesParams(model ai.Model, context ai.Context, options *ai.SimpleStreamOptions) (responses.ResponseNewParams, error) {
	if options == nil {
		options = &ai.SimpleStreamOptions{}
	}
	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(model.ID),
	}
	if context.SystemPrompt != "" {
		params.Instructions = openai.String(context.SystemPrompt)
	}

	input, err := convertResponsesInput(context.Messages)
	if err != nil {
		return responses.ResponseNewParams{}, err
	}
	params.Input = input

	if len(context.Tools) > 0 {
		params.Tools = convertResponsesTools(context.Tools)
	}
	if options.MaxTokens > 0 {
		params.MaxOutputTokens = openai.Int(int64(options.MaxTokens))
	}
	if options.Temperature != nil {
		params.Temperature = openai.Float(*options.Temperature)
	}
	return params, nil
}

func convertResponsesInput(messages []ai.Message) (responses.ResponseNewParamsInputUnion, error) {
	items := make([]responses.ResponseInputItemUnionParam, 0, len(messages))
	for _, m := range messages {
		switch msg := m.(type) {
		case ai.UserMessage:
			items = append(items, responses.ResponseInputItemUnionParam{
				OfMessage: &responses.EasyInputMessageParam{
					Role:    responses.EasyInputMessageRoleUser,
					Content: responses.EasyInputMessageContentUnionParam{OfString: openai.String(userText(msg.Content))},
				},
			})
		case ai.AssistantMessage:
			// Reuse the assistant's text + tool calls as prior output items.
			textParts := ""
			for _, b := range msg.Content {
				switch block := b.(type) {
				case ai.TextContent:
					textParts += block.Text
				case ai.ToolCall:
					items = append(items, responses.ResponseInputItemUnionParam{
						OfFunctionCall: &responses.ResponseFunctionToolCallParam{
							CallID:    block.ID,
							Name:      block.Name,
							Arguments: string(block.ArgumentsRaw),
						},
					})
				}
			}
			if textParts != "" {
				items = append(items, responses.ResponseInputItemUnionParam{
					OfMessage: &responses.EasyInputMessageParam{
						Role:    responses.EasyInputMessageRoleAssistant,
						Content: responses.EasyInputMessageContentUnionParam{OfString: openai.String(textParts)},
					},
				})
			}
		case ai.ToolResultMessage:
			items = append(items, responses.ResponseInputItemUnionParam{
				OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
					CallID: msg.ToolCallID,
					Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
						OfString: openai.String(toolResultText(msg)),
					},
				},
			})
		}
	}
	return responses.ResponseNewParamsInputUnion{
		OfInputItemList: items,
	}, nil
}

func convertResponsesTools(tools []ai.Tool) []responses.ToolUnionParam {
	out := make([]responses.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		out = append(out, responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        t.Name,
				Description: openai.String(t.Description),
				Parameters:  shared.FunctionParameters(t.Parameters),
			},
		})
	}
	return out
}
