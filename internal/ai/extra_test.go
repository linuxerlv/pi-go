package ai

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// ---- jsonunion round-trip ----

func TestUnmarshalContentBlockAllTypes(t *testing.T) {
	cases := []struct {
		name string
		json string
	}{
		{"text", `{"type":"text","text":"hi"}`},
		{"thinking", `{"type":"thinking","thinking":"hmm"}`},
		{"image", `{"type":"image","data":"AAA","mimeType":"image/png"}`},
		{"toolCall", `{"type":"toolCall","id":"c1","name":"read","arguments":"{\"path\":\"x\"}"}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			b, err := UnmarshalContentBlock([]byte(c.json))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			switch b.(type) {
			case TextContent, ThinkingContent, ImageContent, ToolCall:
			default:
				t.Fatalf("unexpected block type %T for %s", b, c.name)
			}
		})
	}
}

func TestUnmarshalContentBlockUnknownType(t *testing.T) {
	_, err := UnmarshalContentBlock([]byte(`{"type":"bogus"}`))
	if err == nil {
		t.Fatal("expected error for unknown content block type")
	}
	if !strings.Contains(err.Error(), "unknown content block type") {
		t.Fatalf("expected unknown-type error, got: %v", err)
	}
}

func TestToolCallMarshalWithArgumentsRoundTrip(t *testing.T) {
	// ToolCall with Arguments populated but ArgumentsRaw empty must still emit
	// arguments on marshal; round-trip repopulates ArgumentsRaw.
	tc := ToolCall{
		Type:      "toolCall",
		ID:        "c1",
		Name:      "read",
		Arguments: map[string]any{"path": "foo.go"},
	}
	b, err := json.Marshal(tc)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(b), `"path":"foo.go"`) {
		t.Fatalf("arguments missing from marshaled output: %s", b)
	}
	var back ToolCall
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.ID != "c1" || back.Name != "read" {
		t.Fatalf("round-trip lost fields: %+v", back)
	}
	if len(back.ArgumentsRaw) == 0 {
		t.Fatal("ArgumentsRaw should be repopulated on unmarshal")
	}
}

func TestMessageRoundTripUserString(t *testing.T) {
	m := UserMessage{Timestamp: 123, Content: "hello"}
	b, err := MarshalMessage(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(b), `"role":"user"`) || !strings.Contains(string(b), `"hello"`) {
		t.Fatalf("unexpected user json: %s", b)
	}
	back, err := UnmarshalMessage(b)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	u, ok := back.(UserMessage)
	if !ok {
		t.Fatalf("expected UserMessage, got %T", back)
	}
	if u.Content != "hello" {
		t.Fatalf("content mismatch: %v", u.Content)
	}
	if u.Timestamp != 123 {
		t.Fatalf("timestamp mismatch: %d", u.Timestamp)
	}
}

func TestMessageRoundTripUserBlocks(t *testing.T) {
	m := UserMessage{Timestamp: 7, Content: []ContentBlock{
		TextContent{Type: "text", Text: "a"},
		ImageContent{Type: "image", Data: "Q", MimeType: "image/png"},
	}}
	b, err := MarshalMessage(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	back, err := UnmarshalMessage(b)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	u, ok := back.(UserMessage)
	if !ok {
		t.Fatalf("expected UserMessage, got %T", back)
	}
	blocks, ok := u.Content.([]ContentBlock)
	if !ok || len(blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %T %v", u.Content, u.Content)
	}
	if _, ok := blocks[1].(ImageContent); !ok {
		t.Fatalf("expected image block, got %T", blocks[1])
	}
}

func TestMessageRoundTripAssistant(t *testing.T) {
	m := AssistantMessage{
		Provider:   "anthropic",
		Model:      "claude-x",
		StopReason: StopToolUse,
		Timestamp:  9,
		Content: []ContentBlock{
			TextContent{Type: "text", Text: "thinking..."},
			ToolCall{Type: "toolCall", ID: "c1", Name: "read", Arguments: map[string]any{"path": "y"}},
		},
	}
	b, err := MarshalMessage(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	back, err := UnmarshalMessage(b)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	a, ok := back.(AssistantMessage)
	if !ok {
		t.Fatalf("expected AssistantMessage, got %T", back)
	}
	if a.StopReason != StopToolUse || len(a.Content) != 2 {
		t.Fatalf("round-trip mismatch: %+v", a)
	}
}

func TestMessageRoundTripToolResult(t *testing.T) {
	m := ToolResultMessage{
		ToolCallID: "c1",
		ToolName:   "read",
		IsError:    true,
		Timestamp:  5,
		Content:    []ContentBlock{TextContent{Type: "text", Text: "boom"}},
		Details:    map[string]any{"code": float64(1)},
	}
	b, err := MarshalMessage(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	back, err := UnmarshalMessage(b)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	tr, ok := back.(ToolResultMessage)
	if !ok {
		t.Fatalf("expected ToolResultMessage, got %T", back)
	}
	if !tr.IsError || tr.ToolCallID != "c1" {
		t.Fatalf("round-trip mismatch: %+v", tr)
	}
	if tr.Details == nil {
		t.Fatal("Details should round-trip")
	}
}

func TestUnmarshalMessageUnknownRoleMessage(t *testing.T) {
	_, err := UnmarshalMessage([]byte(`{"role":"system"}`))
	if err == nil || !strings.Contains(err.Error(), "unknown message role") {
		t.Fatalf("expected unknown-role error, got: %v", err)
	}
}

func TestMarshalMessageUnsupported(t *testing.T) {
	_, err := MarshalMessage(nil)
	if err == nil {
		t.Fatal("expected error for nil message")
	}
}

// ---- validate / coercion ----

func TestValidateCoercionNumberBooleanString(t *testing.T) {
	tool := Tool{
		Name: "x",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"n": map[string]any{"type": "number"},
				"b": map[string]any{"type": "boolean"},
				"s": map[string]any{"type": "string"},
			},
		},
	}
	args, err := ValidateToolArguments(tool, ToolCall{Name: "x", Arguments: map[string]any{
		"n": "3.14",
		"b": "true",
		"s": float64(42),
	}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if f, ok := args["n"].(float64); !ok || f != 3.14 {
		t.Fatalf("n coerce mismatch: %T %v", args["n"], args["n"])
	}
	if b, ok := args["b"].(bool); !ok || !b {
		t.Fatalf("b coerce mismatch: %T %v", args["b"], args["b"])
	}
	if s, ok := args["s"].(string); !ok || s != "42" {
		t.Fatalf("s coerce mismatch: %T %v", args["s"], args["s"])
	}
}

func TestValidateTypeMismatchError(t *testing.T) {
	tool := Tool{
		Name: "x",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"n": map[string]any{"type": "number"},
			},
		},
	}
	_, err := ValidateToolArguments(tool, ToolCall{Name: "x", Arguments: map[string]any{"n": "not-a-number"}})
	if err == nil {
		t.Fatal("expected type-mismatch error for non-numeric string")
	}
	if !strings.Contains(err.Error(), "expected type") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestValidateNestedObjectAndArray(t *testing.T) {
	tool := Tool{
		Name: "x",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"obj": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"k": map[string]any{"type": "integer"},
					},
					"required": []any{"k"},
				},
				"arr": map[string]any{
					"type":  "array",
					"items": map[string]any{"type": "integer"},
				},
			},
		},
	}
	_, err := ValidateToolArguments(tool, ToolCall{Name: "x", Arguments: map[string]any{
		"obj": map[string]any{},
	}})
	if err == nil || !strings.Contains(err.Error(), "missing required property") {
		t.Fatalf("expected nested missing-required error, got: %v", err)
	}
	_, err = ValidateToolArguments(tool, ToolCall{Name: "x", Arguments: map[string]any{
		"obj": map[string]any{"k": "1"},
		"arr": []any{"abc"},
	}})
	if err == nil {
		t.Fatal("expected array element type-mismatch error")
	}
}

func TestValidateNilSchemaPassesThrough(t *testing.T) {
	tool := Tool{Name: "x"}
	args, err := ValidateToolArguments(tool, ToolCall{Name: "x", Arguments: map[string]any{"a": float64(1)}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if args["a"] != float64(1) {
		t.Fatalf("args should pass through unchanged: %v", args)
	}
}

func TestValidateNilArgumentsReturnsEmptyMap(t *testing.T) {
	tool := Tool{Name: "x"}
	args, err := ValidateToolArguments(tool, ToolCall{Name: "x"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if args == nil || len(args) != 0 {
		t.Fatalf("expected non-nil empty map, got %v", args)
	}
}

func TestValidateDoesNotMutateInput(t *testing.T) {
	tool := Tool{
		Name: "x",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"n": map[string]any{"type": "integer"},
			},
		},
	}
	input := map[string]any{"n": "5"}
	_, err := ValidateToolArguments(tool, ToolCall{Name: "x", Arguments: input})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v, ok := input["n"].(string); !ok || v != "5" {
		t.Fatalf("input was mutated: %v", input["n"])
	}
}

// ---- MustJSON ----

func TestMustJSON(t *testing.T) {
	out := MustJSON(map[string]any{"k": "v"})
	if !strings.Contains(out, `"k": "v"`) {
		t.Fatalf("unexpected MustJSON output: %s", out)
	}
	// An unmarshalable value falls back to %+v without panicking.
	_ = MustJSON(make(chan int))
}

// ---- EventStream edge cases ----

func TestEventStreamEndWithValue(t *testing.T) {
	s := NewEventStream[int, int](
		func(e int) bool { return false },
		func(e int) int { return e },
	)
	go func() {
		s.Push(1)
		r := 99
		s.End(&r)
	}()
	var got []int
	for ev := range s.Range {
		got = append(got, ev)
	}
	if len(got) != 1 || got[0] != 1 {
		t.Fatalf("expected [1], got %v", got)
	}
	if r := <-s.Result(); r != 99 {
		t.Fatalf("expected result 99, got %v", r)
	}
}

func TestEventStreamPushAfterDoneIsNoOp(t *testing.T) {
	s := NewEventStream[int, int](
		func(e int) bool { return e == -1 },
		func(e int) int { return e },
	)
	go func() {
		s.Push(-1)
		s.Push(5)
	}()
	count := 0
	for range s.Range {
		count++
	}
	if count != 1 {
		t.Fatalf("expected 1 event after done, got %d", count)
	}
	if r := <-s.Result(); r != -1 {
		t.Fatalf("expected result -1, got %v", r)
	}
}

func TestEventStreamRangeStopsOnBreak(t *testing.T) {
	s := NewEventStream[int, int](
		func(e int) bool { return e == -1 },
		func(e int) int { return e },
	)
	go func() {
		s.Push(1)
		s.Push(2)
		s.Push(-1)
	}()
	var got []int
	for ev := range s.Range {
		got = append(got, ev)
		if len(got) == 1 {
			break
		}
	}
	if len(got) != 1 {
		t.Fatalf("expected early break at 1 event, got %d", len(got))
	}
	if r := <-s.Result(); r != -1 {
		t.Fatalf("expected result -1, got %v", r)
	}
}

var _ = errors.New
