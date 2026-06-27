package tui

import (
	"bytes"
	"strings"
	"testing"
)

func TestRenderDifferentialOnlyWritesChangedRows(t *testing.T) {
	var buf bytes.Buffer
	term := NewTerminal(&buf)

	// First render: all rows are new, all written.
	term.RenderDifferential([]string{"a", "b", "c"})
	first := buf.String()
	if !strings.Contains(first, "a") || !strings.Contains(first, "c") {
		t.Fatalf("first render should contain all rows: %q", first)
	}

	// Second render: only row 1 changes ("a" -> "A"). The diff should rewrite
	// only row 1, not rows 2/3.
	buf.Reset()
	term.RenderDifferential([]string{"A", "b", "c"})
	second := buf.String()
	// Must contain the new "A" row.
	if !strings.Contains(second, "A") {
		t.Fatalf("diff should contain changed row A: %q", second)
	}
	// Unchanged rows "b" and "c" must NOT be rewritten (they may only appear
	// as a trailing cursor move to the last row, never as cleared+written text).
	if strings.Contains(second, "\x1b[2;1H\x1b[K") {
		t.Fatalf("diff should not clear+rewrite row 2: %q", second)
	}
	// Row 1 should be cleared and rewritten.
	if !strings.Contains(second, "\x1b[1;1H\x1b[KA") {
		t.Fatalf("diff should clear+rewrite row 1 with A: %q", second)
	}
}

func TestRenderDifferentialShrinks(t *testing.T) {
	var buf bytes.Buffer
	term := NewTerminal(&buf)
	term.RenderDifferential([]string{"x", "y", "z"})
	buf.Reset()
	// Shrink to one row; rows 2 and 3 should be cleared.
	term.RenderDifferential([]string{"x"})
	out := buf.String()
	// Row 1 unchanged (x) -> skipped. Rows 2,3 cleared.
	if strings.Contains(out, "2;1") == false && strings.Contains(out, "3;1") == false {
		// At least one of rows 2/3 should be targeted for clearing.
		t.Fatalf("expected rows 2/3 to be cleared on shrink: %q", out)
	}
}

func TestFrameAppendAndSet(t *testing.T) {
	f := NewFrame()
	f.Append("a")
	f.Appendf("b%d", 2)
	f.Set(4, "e")
	rows := f.Rows()
	if len(rows) != 5 || rows[0] != "a" || rows[1] != "b2" || rows[4] != "e" {
		t.Fatalf("unexpected rows: %v", rows)
	}
}
