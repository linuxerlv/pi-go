package ai

import (
	"testing"
)

func TestEventStreamDeliversEventsAndResult(t *testing.T) {
	s := NewEventStream[int, int](
		func(e int) bool { return e == -1 },
		func(e int) int { return e },
	)

	go func() {
		s.Push(1)
		s.Push(2)
		s.Push(3)
		s.Push(-1) // terminal
	}()

	var got []int
	for ev := range s.Range {
		got = append(got, ev)
	}

	if len(got) != 4 {
		t.Fatalf("expected 4 events, got %d (%v)", len(got), got)
	}
	if got[0] != 1 || got[3] != -1 {
		t.Fatalf("unexpected order: %v", got)
	}

	r := <-s.Result()
	if r != -1 {
		t.Fatalf("expected result -1, got %d", r)
	}
}

func TestEventStreamEndWithoutTerminal(t *testing.T) {
	s := NewEventStream[int, int](
		func(e int) bool { return e == -1 },
		func(e int) int { return e },
	)

	go func() {
		s.Push(1)
		s.Push(2)
		s.End(nil) // no terminal event, no result
	}()

	var got []int
	for ev := range s.Range {
		got = append(got, ev)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 events, got %d", len(got))
	}

	// Result channel closed without a value.
	if _, ok := <-s.Result(); ok {
		t.Fatalf("expected result channel closed without value")
	}
}

func TestValidateToolArgumentsRequiredAndCoercion(t *testing.T) {
	tool := Tool{
		Name: "read",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":   map[string]any{"type": "string"},
				"offset": map[string]any{"type": "integer"},
			},
			"required": []any{"path"},
		},
	}

	// Missing required field.
	_, err := ValidateToolArguments(tool, ToolCall{Name: "read", Arguments: map[string]any{}})
	if err == nil {
		t.Fatal("expected error for missing required field")
	}

	// String number coerced to integer.
	args, err := ValidateToolArguments(tool, ToolCall{
		Name:      "read",
		Arguments: map[string]any{"path": "/tmp/x", "offset": "42"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if off, ok := args["offset"].(int64); !ok || off != 42 {
		t.Fatalf("expected offset int64 42, got %T %v", args["offset"], args["offset"])
	}
}
