package provider

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// finalizedToolCalls runs finalize on the translator and returns the ToolCall
// blocks from its snapshot, mirroring how the existing openai_test inspects
// translator state (without consuming the event stream).
func finalizedToolCalls(tr *anthropicTranslator, stream ai.AssistantMessageEventStream) []ai.ToolCall {
	tr.finalize(stream)
	var out []ai.ToolCall
	for _, b := range tr.snapshot().Content {
		if tc, ok := b.(ai.ToolCall); ok {
			out = append(out, tc)
		}
	}
	return out
}

// TestAnthropicStartInputAuthoritative covers the gateway quirk where a
// content_block_start for tool_use carries the FULL input object (len>0) and the
// server also streams input_json_delta afterward. The start input must win;
// deltas must be ignored (previously the first delta discarded start's args).
func TestAnthropicStartInputAuthoritative(t *testing.T) {
	model := ai.Model{ID: "claude-x", API: ai.APIAnthropicMessages, Provider: "anthropic"}
	tr := newAnthropicTranslator(model)
	stream := ai.NewAssistantMessageEventStream()

	startInput := map[string]any{"path": "foo.go", "limit": float64(10)}

	// content_block_start: tool_use with a REAL (non-empty) input.
	tr.handle(anthropic.MessageStreamEventUnion{
		Type:  "content_block_start",
		Index: 0,
		ContentBlock: anthropic.ContentBlockStartEventContentBlockUnion{
			Type:  "tool_use",
			ID:    "call_1",
			Name:  "read",
			Input: startInput,
		},
	}, stream)

	// input_json_delta arriving AFTER start already had real args. With the fix
	// these are ignored (start is authoritative); previously they would clear
	// start's args and append fragments, corrupting them.
	tr.handle(anthropic.MessageStreamEventUnion{
		Type:  "content_block_delta",
		Index: 0,
		Delta: anthropic.MessageStreamEventUnionDelta{
			Type:        "input_json_delta",
			PartialJSON: `{"path":"WRONG"}`,
		},
	}, stream)

	// content_block_stop finalizes the tool call.
	tr.handle(anthropic.MessageStreamEventUnion{
		Type:  "content_block_stop",
		Index: 0,
	}, stream)

	tcs := finalizedToolCalls(tr, stream)
	if len(tcs) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(tcs))
	}
	got := tcs[0].Arguments
	if got["path"] != "foo.go" {
		t.Fatalf("start input should be authoritative; got path=%v want foo.go (args=%v)", got["path"], got)
	}
	if got["limit"] != float64(10) {
		t.Fatalf("start input should be authoritative; got limit=%v want 10 (args=%v)", got["limit"], got)
	}
}

// TestAnthropicDeltaBuildsArgsFromEmptyStart is the standard-protocol path:
// content_block_start carries an empty {} placeholder (no real args), so deltas
// assemble the arguments. This must still work after the fix.
func TestAnthropicDeltaBuildsArgsFromEmptyStart(t *testing.T) {
	model := ai.Model{ID: "claude-x", API: ai.APIAnthropicMessages, Provider: "anthropic"}
	tr := newAnthropicTranslator(model)
	stream := ai.NewAssistantMessageEventStream()

	// Standard protocol: start with empty {} input (len==0 -> not captured).
	tr.handle(anthropic.MessageStreamEventUnion{
		Type:  "content_block_start",
		Index: 0,
		ContentBlock: anthropic.ContentBlockStartEventContentBlockUnion{
			Type:  "tool_use",
			ID:    "call_1",
			Name:  "read",
			Input: map[string]any{}, // empty placeholder
		},
	}, stream)

	tr.handle(anthropic.MessageStreamEventUnion{
		Type:  "content_block_delta",
		Index: 0,
		Delta: anthropic.MessageStreamEventUnionDelta{
			Type:        "input_json_delta",
			PartialJSON: `{"path":"bar.go"}`,
		},
	}, stream)

	tr.handle(anthropic.MessageStreamEventUnion{
		Type:  "content_block_stop",
		Index: 0,
	}, stream)

	tcs := finalizedToolCalls(tr, stream)
	if len(tcs) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(tcs))
	}
	if tcs[0].Arguments["path"] != "bar.go" {
		t.Fatalf("deltas should assemble args from empty start; got %v", tcs[0].Arguments)
	}
}
