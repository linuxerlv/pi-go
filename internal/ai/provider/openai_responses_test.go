package provider

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go/v3/responses"

	"github.com/linuxerlv/pi-go/internal/ai"
)

func TestResponsesTranslatorTextAndToolCall(t *testing.T) {
	model := ai.Model{ID: "gpt-4o", API: ai.APIOpenAIResponses, Provider: "openai-responses"}
	tr := newResponsesTranslator(model)
	stream := ai.NewAssistantMessageEventStream()

	// Text deltas.
	tr.handle(responses.ResponseStreamEventUnion{Type: "response.output_text.delta", Delta: "Hello"}, stream)
	tr.handle(responses.ResponseStreamEventUnion{Type: "response.output_text.delta", Delta: " world"}, stream)

	// Function call: output_item.added then argument deltas.
	tr.handle(responses.ResponseStreamEventUnion{
		Type:        "response.output_item.added",
		OutputIndex: 1,
		Item: responses.ResponseOutputItemUnion{Type: "function_call", CallID: "call_1", Name: "read"},
	}, stream)
	tr.handle(responses.ResponseStreamEventUnion{
		Type: "response.function_call_arguments.delta", OutputIndex: 1, Arguments: `{"path":"go`,
	}, stream)
	tr.handle(responses.ResponseStreamEventUnion{
		Type: "response.function_call_arguments.delta", OutputIndex: 1, Arguments: `.mod"}`,
	}, stream)
	tr.handle(responses.ResponseStreamEventUnion{
		Type: "response.function_call_arguments.done", OutputIndex: 1, Arguments: `{"path":"go.mod"}`,
	}, stream)

	// Completed with tool calls -> stopReason toolUse.
	tr.handle(responses.ResponseStreamEventUnion{Type: "response.completed", Response: responses.Response{
		Status: "completed",
	}}, stream)

	tr.finalize(stream)

	msg := tr.snapshot()
	if msg.StopReason != ai.StopStop {
		// "completed" maps to stop; toolUse is inferred only when no completed status.
		t.Fatalf("expected stopReason stop (completed), got %s", msg.StopReason)
	}
	if len(msg.Content) != 2 {
		t.Fatalf("expected 2 content blocks (text + tool call), got %d", len(msg.Content))
	}
	if tc, ok := msg.Content[1].(ai.ToolCall); !ok || tc.Name != "read" || tc.ID != "call_1" {
		t.Fatalf("unexpected tool call: %+v", msg.Content[1])
	} else if tc.Arguments["path"] != "go.mod" {
		t.Fatalf("expected path=go.mod, got %v", tc.Arguments["path"])
	}
}

func TestResponsesMapStatus(t *testing.T) {
	cases := map[string]ai.StopReason{
		"completed":   ai.StopStop,
		"incomplete":  ai.StopLength,
		"failed":      ai.StopError,
		"cancelled":   ai.StopError,
		"":            "",
	}
	for in, want := range cases {
		if got := mapResponsesStatus(in); got != want {
			t.Errorf("mapResponsesStatus(%q) = %q, want %q", in, got, want)
		}
	}
}

// Verify arguments raw JSON round-trips.
func TestResponsesToolCallArgsRaw(t *testing.T) {
	model := ai.Model{ID: "gpt-4o", API: ai.APIOpenAIResponses}
	tr := newResponsesTranslator(model)
	stream := ai.NewAssistantMessageEventStream()
	tr.handle(responses.ResponseStreamEventUnion{
		Type: "response.output_item.added", OutputIndex: 0,
		Item: responses.ResponseOutputItemUnion{Type: "function_call", CallID: "c", Name: "f"},
	}, stream)
	tr.handle(responses.ResponseStreamEventUnion{
		Type: "response.function_call_arguments.delta", OutputIndex: 0, Arguments: `{"a":1}`,
	}, stream)
	tr.finalize(stream)
	msg := tr.snapshot()
	tc, ok := msg.Content[0].(ai.ToolCall)
	if !ok {
		t.Fatalf("expected tool call, got %T", msg.Content[0])
	}
	var m map[string]any
	if err := json.Unmarshal(tc.ArgumentsRaw, &m); err != nil {
		t.Fatalf("raw args invalid: %v", err)
	}
	if m["a"].(float64) != 1 {
		t.Fatalf("expected a=1, got %v", m["a"])
	}
}
