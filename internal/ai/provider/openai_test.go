package provider

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go/v3"

	"github.com/linuxerlv/pi-go/internal/ai"
)

func TestOpenAITranslatorTextAndToolCall(t *testing.T) {
	model := ai.Model{ID: "gpt-4o-mini", API: ai.APIOpenAICompletions, Provider: "openai"}
	tr := newOpenAITranslator(model)
	stream := ai.NewAssistantMessageEventStream()

	// Simulate a streamed response: text delta, then a tool call split across
	// two argument fragments, then finish_reason=tool_calls.
	tr.handle(openai.ChatCompletionChunk{
		ID: "chat_1",
		Choices: []openai.ChatCompletionChunkChoice{{
			Index: 0,
			Delta: openai.ChatCompletionChunkChoiceDelta{Content: "Hello"},
		}},
	}, stream)

	tr.handle(openai.ChatCompletionChunk{
		ID: "chat_1",
		Choices: []openai.ChatCompletionChunkChoice{{
			Index: 0,
			Delta: openai.ChatCompletionChunkChoiceDelta{
				ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{{
					Index: 0,
					ID:    "call_1",
					Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
						Name: "read",
					},
				}},
			},
		}},
	}, stream)

	tr.handle(openai.ChatCompletionChunk{
		ID: "chat_1",
		Choices: []openai.ChatCompletionChunkChoice{{
			Index: 0,
			Delta: openai.ChatCompletionChunkChoiceDelta{
				ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{{
					Index: 0,
					Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
						Arguments: `{"path":"go`,
					},
				}},
			},
		}},
	}, stream)

	tr.handle(openai.ChatCompletionChunk{
		ID: "chat_1",
		Choices: []openai.ChatCompletionChunkChoice{{
			Index: 0,
			Delta: openai.ChatCompletionChunkChoiceDelta{
				ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{{
					Index: 0,
					Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
						Arguments: `.mod"}`,
					},
				}},
			},
		}},
	}, stream)

	tr.handle(openai.ChatCompletionChunk{
		ID: "chat_1",
		Choices: []openai.ChatCompletionChunkChoice{{
			Index:        0,
			Delta:        openai.ChatCompletionChunkChoiceDelta{},
			FinishReason: "tool_calls",
		}},
	}, stream)

	// finalize decodes the accumulated tool-call arguments.
	tr.finalize(stream)

	msg := tr.snapshot()
	if msg.StopReason != ai.StopToolUse {
		t.Fatalf("expected stopReason toolUse, got %s", msg.StopReason)
	}

	// Expect two content blocks: text + tool call.
	if len(msg.Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(msg.Content))
	}
	tc, ok := msg.Content[1].(ai.ToolCall)
	if !ok {
		t.Fatalf("expected tool call at index 1, got %T", msg.Content[1])
	}
	if tc.Name != "read" || tc.ID != "call_1" {
		t.Fatalf("unexpected tool call: %+v", tc)
	}
	if tc.Arguments["path"] != "go.mod" {
		t.Fatalf("expected path=go.mod, got %v", tc.Arguments["path"])
	}

	// Verify the accumulated raw JSON round-trips.
	var decoded map[string]any
	if err := json.Unmarshal(tc.ArgumentsRaw, &decoded); err != nil {
		t.Fatalf("arguments raw JSON invalid: %v", err)
	}
	if decoded["path"] != "go.mod" {
		t.Fatalf("raw JSON path mismatch: %v", decoded["path"])
	}
}

func TestOpenAIMapStopReason(t *testing.T) {
	cases := map[string]ai.StopReason{
		"stop":          ai.StopStop,
		"length":        ai.StopLength,
		"tool_calls":    ai.StopToolUse,
		"function_call": ai.StopToolUse,
		"":              "",
		"content_filter": ai.StopStop,
	}
	for in, want := range cases {
		if got := mapOpenAIStopReason(in); got != want {
			t.Errorf("mapOpenAIStopReason(%q) = %q, want %q", in, got, want)
		}
	}
}
