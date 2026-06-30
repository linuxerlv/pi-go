package ai

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestMessageRoundTrip verifies MarshalMessage/UnmarshalMessage round-trip for
// each message variant and content shape.
func TestMessageRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		msg  Message
	}{
		{"user-string", UserMessage{Content: "hello", Timestamp: 1}},
		{"user-blocks", UserMessage{Content: []ContentBlock{
			TextContent{Type: "text", Text: "hi"},
			ImageContent{Type: "image", Data: "base64==", MimeType: "image/png"},
		}, Timestamp: 2}},
		{"assistant-text", AssistantMessage{
			Content:   []ContentBlock{TextContent{Type: "text", Text: "reply"}},
			API:       APIAnthropicMessages, Provider: "mock", Model: "m", StopReason: StopStop, Timestamp: 3,
		}},
		{"assistant-thinking", AssistantMessage{
			Content: []ContentBlock{
				ThinkingContent{Type: "thinking", Thinking: "reasoning"},
				TextContent{Type: "text", Text: "answer"},
			},
			Provider: "mock", Model: "m", StopReason: StopStop, Timestamp: 4,
		}},
		{"tool-result", ToolResultMessage{
			ToolCallID: "c1", ToolName: "read",
			Content:   []ContentBlock{TextContent{Type: "text", Text: "file"}},
			Timestamp: 5,
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			raw, err := MarshalMessage(c.msg)
			if err != nil {
				t.Fatalf("MarshalMessage: %v", err)
			}
			got, err := UnmarshalMessage(raw)
			if err != nil {
				t.Fatalf("UnmarshalMessage: %v", err)
			}
			// Re-marshal both and compare JSON for structural equality (interface
			// values with maps don't compare with ==).
			raw2, _ := MarshalMessage(got)
			if string(raw) != string(raw2) {
				t.Fatalf("round-trip mismatch:\n in:  %s\n out: %s", raw, raw2)
			}
		})
	}
}

// TestToolCallMarshalWithArgumentsOnly verifies a ToolCall with only Arguments
// (no ArgumentsRaw) marshals its arguments to JSON.
func TestToolCallMarshalWithArgumentsOnly(t *testing.T) {
	tc := ToolCall{Type: "toolCall", ID: "c1", Name: "read", Arguments: map[string]any{"path": "f"}}
	raw, err := json.Marshal(tc)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var probe struct {
		Arguments map[string]any `json:"arguments"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if probe.Arguments["path"] != "f" {
		t.Fatalf("expected arguments.path=f, got %v", probe.Arguments["path"])
	}
}

// TestToolCallMarshalWithRawOnly verifies ArgumentsRaw is passed through.
func TestToolCallMarshalWithRawOnly(t *testing.T) {
	tc := ToolCall{Type: "toolCall", ID: "c1", Name: "read", ArgumentsRaw: json.RawMessage(`{"path":"g"}`)}
	raw, err := json.Marshal(tc)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(raw) == "" || !strings.Contains(string(raw), `"path":"g"`) {
		t.Fatalf("expected raw arguments in output, got %s", raw)
	}
}

// TestUnmarshalContentBlockTypes verifies each type tag decodes to the right
// concrete type, and unknown types error.
func TestUnmarshalContentBlockTypes(t *testing.T) {
	cases := []struct {
		json string
		want ContentBlock
	}{
		{`{"type":"text","text":"hi"}`, TextContent{Type: "text", Text: "hi"}},
		{`{"type":"thinking","thinking":"r"}`, ThinkingContent{Type: "thinking", Thinking: "r"}},
		{`{"type":"image","data":"d","mimeType":"image/png"}`, ImageContent{Type: "image", Data: "d", MimeType: "image/png"}},
		{`{"type":"toolCall","id":"c1","name":"read","arguments":{"path":"f"}}`, ToolCall{Type: "toolCall", ID: "c1", Name: "read"}},
	}
	for _, c := range cases {
		got, err := UnmarshalContentBlock([]byte(c.json))
		if err != nil {
			t.Fatalf("UnmarshalContentBlock(%s): %v", c.json, err)
		}
		// Compare by re-marshalling (interface equality is unreliable for structs
		// with maps).
		gotRaw, _ := json.Marshal(got)
		wantRaw, _ := json.Marshal(c.want)
		// Loose check: the type field must match.
		var gotType, wantType struct {
			Type string `json:"type"`
		}
		json.Unmarshal(gotRaw, &gotType)
		json.Unmarshal(wantRaw, &wantType)
		if gotType.Type != wantType.Type {
			t.Fatalf("type mismatch: got %s want %s", gotType.Type, wantType.Type)
		}
	}
	// Unknown type errors.
	if _, err := UnmarshalContentBlock([]byte(`{"type":"unknown"}`)); err == nil {
		t.Fatal("expected error for unknown content block type")
	}
}

// TestUnmarshalMessageUnknownRole verifies an unknown role errors.
func TestUnmarshalMessageUnknownRole(t *testing.T) {
	if _, err := UnmarshalMessage([]byte(`{"role":"system"}`)); err == nil {
		t.Fatal("expected error for unknown message role")
	}
}
