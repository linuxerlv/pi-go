package provider

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// TestAnthropicGatewayMissingStopEvents covers the quirk recorded in project
// memory anthropic-gateway-stream-quirk: some local gateways omit
// content_block_stop AND message_stop. The translator's finalize step must
// still decode the accumulated tool-call args from input_json_delta fragments.
// (The existing tests always send content_block_stop; this one does not.)
func TestAnthropicGatewayMissingStopEvents(t *testing.T) {
	model := ai.Model{ID: "claude-x", API: ai.APIAnthropicMessages, Provider: "anthropic"}
	tr := newAnthropicTranslator(model)
	stream := ai.NewAssistantMessageEventStream()

	// tool_use start with empty placeholder (standard protocol).
	tr.handle(anthropic.MessageStreamEventUnion{
		Type:  "content_block_start",
		Index: 0,
		ContentBlock: anthropic.ContentBlockStartEventContentBlockUnion{
			Type: "tool_use",
			ID:   "call_1",
			Name: "write",
			Input: map[string]any{},
		},
	}, stream)

	// Stream args in two fragments — but NEVER send content_block_stop or
	// message_stop, simulating a truncated gateway stream.
	tr.handle(anthropic.MessageStreamEventUnion{
		Type:  "content_block_delta",
		Index: 0,
		Delta: anthropic.MessageStreamEventUnionDelta{
			Type:        "input_json_delta",
			PartialJSON: `{"path":"out`,
		},
	}, stream)
	tr.handle(anthropic.MessageStreamEventUnion{
		Type:  "content_block_delta",
		Index: 0,
		Delta: anthropic.MessageStreamEventUnionDelta{
			Type:        "input_json_delta",
			PartialJSON: `.txt","n":3}`,
		},
	}, stream)

	// finalize must decode the accumulated raw JSON into Arguments.
	tr.finalize(stream)
	tcs := finalizedToolCalls(tr, stream)
	if len(tcs) != 1 {
		t.Fatalf("expected 1 tool call after finalize, got %d", len(tcs))
	}
	if tcs[0].Arguments["path"] != "out.txt" {
		t.Fatalf("finalize should decode args; got path=%v (args=%v)", tcs[0].Arguments["path"], tcs[0].Arguments)
	}
	if tcs[0].Arguments["n"] != float64(3) {
		t.Fatalf("finalize should decode n=3; got %v", tcs[0].Arguments["n"])
	}
}

// TestAnthropicTextStreamingAndTerminal exercises the text path end-to-end:
// message_start seeds usage, text_delta accumulates, message_stop emits a
// terminal DoneEvent with stopReason defaulting to "stop".
func TestAnthropicTextStreamingAndTerminal(t *testing.T) {
	model := ai.Model{ID: "claude-x", API: ai.APIAnthropicMessages, Provider: "anthropic"}
	tr := newAnthropicTranslator(model)
	stream := ai.NewAssistantMessageEventStream()

	tr.handle(anthropic.MessageStreamEventUnion{
		Type: "message_start",
		Message: anthropic.Message{
			ID: "msg_1",
			Usage: anthropic.Usage{
				InputTokens:  10,
				OutputTokens: 0,
			},
		},
	}, stream)

	tr.handle(anthropic.MessageStreamEventUnion{
		Type:  "content_block_start",
		Index: 0,
		ContentBlock: anthropic.ContentBlockStartEventContentBlockUnion{
			Type: "text",
		},
	}, stream)

	tr.handle(anthropic.MessageStreamEventUnion{
		Type:  "content_block_delta",
		Index: 0,
		Delta: anthropic.MessageStreamEventUnionDelta{
			Type: "text_delta",
			Text: "Hello ",
		},
	}, stream)
	tr.handle(anthropic.MessageStreamEventUnion{
		Type:  "content_block_delta",
		Index: 0,
		Delta: anthropic.MessageStreamEventUnionDelta{
			Type: "text_delta",
			Text: "world",
		},
	}, stream)

	tr.handle(anthropic.MessageStreamEventUnion{
		Type:  "content_block_stop",
		Index: 0,
	}, stream)

	// message_stop is terminal: it pushes a DoneEvent and sets terminalEmitted.
	tr.handle(anthropic.MessageStreamEventUnion{
		Type: "message_stop",
	}, stream)

	// Drain the event stream and look for the terminal DoneEvent + accumulated text.
	var gotText string
	var doneReason ai.StopReason
	for ev := range stream.Range {
		switch e := ev.(type) {
		case ai.TextDeltaEvent:
			gotText += e.Delta
		case ai.DoneEvent:
			doneReason = e.Reason
			if len(e.Message.Content) == 0 {
				t.Fatal("DoneEvent message should have content")
			}
			if tc, ok := e.Message.Content[0].(ai.TextContent); ok {
				if tc.Text != "Hello world" {
					t.Fatalf("expected accumulated text 'Hello world', got %q", tc.Text)
				}
			}
		}
	}
	if gotText != "Hello world" {
		t.Fatalf("expected delta text 'Hello world', got %q", gotText)
	}
	if doneReason != ai.StopStop {
		t.Fatalf("expected stopReason stop, got %s", doneReason)
	}
	if !tr.terminalEmitted() {
		t.Fatal("terminalEmitted should be true after message_stop")
	}
}

// TestAnthropicMessageDeltaStopReason ensures message_delta updates the stop
// reason (e.g. tool_use) before message_stop finalizes it.
func TestAnthropicMessageDeltaStopReason(t *testing.T) {
	model := ai.Model{ID: "claude-x", API: ai.APIAnthropicMessages, Provider: "anthropic"}
	tr := newAnthropicTranslator(model)
	stream := ai.NewAssistantMessageEventStream()

	tr.handle(anthropic.MessageStreamEventUnion{Type: "message_start"}, stream)
	tr.handle(anthropic.MessageStreamEventUnion{
		Type: "message_delta",
		Delta: anthropic.MessageStreamEventUnionDelta{
			StopReason: "tool_use",
		},
	}, stream)
	tr.handle(anthropic.MessageStreamEventUnion{Type: "message_stop"}, stream)

	var done ai.DoneEvent
	for ev := range stream.Range {
		if d, ok := ev.(ai.DoneEvent); ok {
			done = d
		}
	}
	if done.Reason != ai.StopToolUse {
		t.Fatalf("expected stopReason toolUse from message_delta, got %s", done.Reason)
	}
}

func TestMapStopReason(t *testing.T) {
	cases := map[string]ai.StopReason{
		"end_turn":      ai.StopStop,
		"stop_sequence": ai.StopStop,
		"max_tokens":    ai.StopLength,
		"tool_use":      ai.StopToolUse,
		"":              "",
		"unknown":       ai.StopStop, // default
	}
	for in, want := range cases {
		if got := mapStopReason(in); got != want {
			t.Errorf("mapStopReason(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestAnthropicConvertMessages covers the message→params conversion for all
// three message kinds, including tool results and image user content.
func TestAnthropicConvertMessages(t *testing.T) {
	msgs := []ai.Message{
		ai.UserMessage{Content: "hi"},
		ai.UserMessage{Content: []ai.ContentBlock{
			ai.TextContent{Type: "text", Text: "look"},
			ai.ImageContent{Type: "image", Data: "AAA", MimeType: "image/png"},
		}},
		ai.AssistantMessage{Content: []ai.ContentBlock{
			ai.TextContent{Type: "text", Text: "ok"},
		}},
		ai.ToolResultMessage{
			ToolCallID: "c1",
			ToolName:   "read",
			Content:    []ai.ContentBlock{ai.TextContent{Type: "text", Text: "file body"}},
		},
	}
	out, err := convertMessages(msgs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != 4 {
		t.Fatalf("expected 4 params, got %d", len(out))
	}
}

func TestAnthropicConvertUserContentUnsupported(t *testing.T) {
	_, err := convertUserContent(12345)
	if err == nil {
		t.Fatal("expected error for unsupported user content type")
	}
}

func TestAnthropicExtraSchemaFieldsAndRequired(t *testing.T) {
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"a": map[string]any{"type": "string"}},
		"required":   []any{"a", "b"},
		"additionalProperties": false,
	}
	req := requiredFromSchema(schema)
	if len(req) != 2 || req[0] != "a" || req[1] != "b" {
		t.Fatalf("required mismatch: %v", req)
	}
	extra := extraSchemaFields(schema)
	if _, ok := extra["additionalProperties"]; !ok {
		t.Fatalf("extra should preserve additionalProperties: %v", extra)
	}
	if _, ok := extra["type"]; ok {
		t.Fatal("extra should not include type")
	}
	if extra := extraSchemaFields(map[string]any{"type": "object"}); extra != nil {
		t.Fatalf("empty extra should be nil, got %v", extra)
	}
}

func TestAnthropicBuildParamsDefaults(t *testing.T) {
	// No MaxTokens in options or model -> defaults to 4096.
	model := ai.Model{ID: "claude-x"}
	params, err := buildAnthropicParams(model, ai.Context{SystemPrompt: "sys"}, &ai.SimpleStreamOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.MaxTokens != 4096 {
		t.Fatalf("expected default maxTokens 4096, got %d", params.MaxTokens)
	}
	if params.Model != "claude-x" {
		t.Fatalf("expected model id, got %s", params.Model)
	}
	// System prompt should populate the system block.
	if len(params.System) == 0 || params.System[0].Text != "sys" {
		t.Fatalf("system prompt not set: %v", params.System)
	}
}

func TestAnthropicBuildParamsFromModelMaxTokens(t *testing.T) {
	model := ai.Model{ID: "claude-x", MaxTokens: 8192}
	params, err := buildAnthropicParams(model, ai.Context{}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.MaxTokens != 8192 {
		t.Fatalf("expected model maxTokens 8192, got %d", params.MaxTokens)
	}
}
