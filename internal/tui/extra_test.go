package tui

import (
	"strings"
	"testing"
)

func TestTerminalPrimitives(t *testing.T) {
	var buf strings.Builder
	term := NewTerminal(&buf)

	term.ClearScreen()
	term.MoveTo(3, 5)
	term.ClearLine()
	term.HideCursor()
	term.ShowCursor()
	term.WriteString("raw")

	out := buf.String()
	for _, want := range []string{"\033[2J\033[H", "\033[3;5H", "\033[K", "\033[?25l", "\033[?25h", "raw"} {
		if !strings.Contains(out, want) {
			t.Fatalf("expected %q in output: %q", want, out)
		}
	}
}

func TestTerminalWidthHeight(t *testing.T) {
	term := NewTerminal(nil)
	if term.Width() != 80 || term.Height() != 24 {
		t.Fatalf("expected default 80x24, got %dx%d", term.Width(), term.Height())
	}
}

func TestTerminalIsTTYFalseForNonFile(t *testing.T) {
	term := NewTerminal(&strings.Builder{})
	if term.IsTTY() {
		t.Fatal("IsTTY should be false for a non-file writer")
	}
}

func TestRenderDifferentialResetRedrawsAll(t *testing.T) {
	var buf strings.Builder
	term := NewTerminal(&buf)
	term.RenderDifferential([]string{"a", "b"})
	buf.Reset()
	term.Reset()
	term.RenderDifferential([]string{"a", "b"})
	out := buf.String()
	if !strings.Contains(out, "\x1b[1;1H\x1b[Ka") {
		t.Fatalf("reset should force redraw of row 1: %q", out)
	}
	if !strings.Contains(out, "\x1b[2;1H\x1b[Kb") {
		t.Fatalf("reset should force redraw of row 2: %q", out)
	}
}

func TestRenderDifferentialUnchangedSkipped(t *testing.T) {
	var buf strings.Builder
	term := NewTerminal(&buf)
	term.RenderDifferential([]string{"a", "b"})
	buf.Reset()
	term.RenderDifferential([]string{"a", "b"})
	out := buf.String()
	if strings.Contains(out, "\x1b[K") {
		t.Fatalf("identical re-render should not clear any row: %q", out)
	}
}

func TestFrameStringAndRows(t *testing.T) {
	f := NewFrame()
	f.Append("a")
	f.Append("b")
	if got := f.String(); got != "a\nb" {
		t.Fatalf("String mismatch: %q", got)
	}
	// Set beyond current length extends with empties.
	f.Set(4, "e")
	rows := f.Rows()
	if len(rows) != 5 || rows[3] != "" || rows[4] != "e" {
		t.Fatalf("Set extension mismatch: %v", rows)
	}
}

func TestItoaViaMoveTo(t *testing.T) {
	var buf strings.Builder
	term := NewTerminal(&buf)
	term.MoveTo(0, 12)
	term.MoveTo(123, 1)
	out := buf.String()
	if !strings.Contains(out, "\033[0;12H") {
		t.Fatalf("itoa(0) path missing: %q", out)
	}
	if !strings.Contains(out, "\033[123;1H") {
		t.Fatalf("multi-digit itoa missing: %q", out)
	}
}
