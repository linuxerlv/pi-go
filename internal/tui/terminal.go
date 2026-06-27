// Package tui is a minimal terminal rendering layer with differential updates.
// It is a small, dependency-free analogue of pi-tui's differential renderer:
// the renderer keeps the previously drawn lines and only emits the ANSI needed
// to update changed lines, avoiding full-screen flicker.
package tui

import (
	"fmt"
	"io"
	"os"
	"strings"
)

// Terminal wraps an output writer with raw ANSI operations.
type Terminal struct {
	w        io.Writer
	width    int
	height   int
	prevRows []string // last rendered row state for differential updates
}

// NewTerminal constructs a Terminal writing to w. Width/height default to 80x24
// (a real TTY size probe could replace this later; kept dependency-free here).
func NewTerminal(w io.Writer) *Terminal {
	return &Terminal{w: w, width: 80, height: 24}
}

// Width returns the terminal width in columns.
func (t *Terminal) Width() int { return t.width }

// Height returns the terminal height in rows.
func (t *Terminal) Height() int { return t.height }

// Write writes raw bytes to the terminal.
func (t *Terminal) Write(b []byte) (int, error) { return t.w.Write(b) }

// WriteString writes a string.
func (t *Terminal) WriteString(s string) (int, error) { return io.WriteString(t.w, s) }

// IsTTY reports whether the output is a character device (a terminal).
func (t *Terminal) IsTTY() bool {
	f, ok := t.w.(*os.File)
	if !ok {
		return false
	}
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return (fi.Mode() & os.ModeCharDevice) != 0
}

// ClearScreen clears the whole screen and moves the cursor home.
func (t *Terminal) ClearScreen() {
	t.WriteString("\033[2J\033[H")
}

// MoveTo moves the cursor to row r (1-indexed), column c (1-indexed).
func (t *Terminal) MoveTo(r, c int) {
	t.WriteString("\033[" + itoa(r) + ";" + itoa(c) + "H")
}

// ClearLine clears the current line from the cursor to the end.
func (t *Terminal) ClearLine() {
	t.WriteString("\033[K")
}

// HideCursor / ShowCursor toggle cursor visibility.
func (t *Terminal) HideCursor() { t.WriteString("\033[?25l") }
func (t *Terminal) ShowCursor() { t.WriteString("\033[?25h") }

// RenderDifferential updates the screen so it shows rows, reusing the previous
// frame: only rows that changed are rewritten. Rows beyond the previous frame
// are appended; rows that shrank clear the leftovers. The cursor is left just
// after the last row.
//
// This is the core of the differential renderer: it computes a row-by-row diff
// against prevRows and emits minimal ANSI updates.
func (t *Terminal) RenderDifferential(rows []string) {
	maxLen := len(t.prevRows)
	if len(rows) > maxLen {
		maxLen = len(rows)
	}
	for i := 0; i < maxLen; i++ {
		var prev, curr string
		if i < len(t.prevRows) {
			prev = t.prevRows[i]
		}
		if i < len(rows) {
			curr = rows[i]
		}
		if prev == curr {
			continue // unchanged row: skip, the heart of differential rendering
		}
		t.MoveTo(i+1, 1)
		t.ClearLine()
		if i < len(rows) {
			t.WriteString(curr)
		}
	}
	t.prevRows = append([]string(nil), rows...)
	// Trim trailing unchanged empties from prevRows so re-renders shrink.
	for len(t.prevRows) > 0 && t.prevRows[len(t.prevRows)-1] == "" {
		t.prevRows = t.prevRows[:len(t.prevRows)-1]
	}
	if len(rows) > 0 {
		t.MoveTo(len(rows), 1)
	}
}

// Reset clears the differential state (e.g. after a full clear) so the next
// render redraws everything.
func (t *Terminal) Reset() { t.prevRows = nil }

// Frame is a mutable buffer of rows being composed for the next render.
type Frame struct {
	rows []string
}

// NewFrame creates an empty frame.
func NewFrame() *Frame { return &Frame{} }

// Append adds a row.
func (f *Frame) Append(row string) { f.rows = append(f.rows, row) }

// Appendf adds a formatted row.
func (f *Frame) Appendf(format string, args ...any) {
	f.rows = append(f.rows, fmt.Sprintf(format, args...))
}

// Set replaces a row at index i, extending the frame if needed.
func (f *Frame) Set(i int, row string) {
	for len(f.rows) <= i {
		f.rows = append(f.rows, "")
	}
	f.rows[i] = row
}

// Rows returns the frame's rows.
func (f *Frame) Rows() []string { return f.rows }

// String joins the rows with newlines.
func (f *Frame) String() string { return strings.Join(f.rows, "\n") }

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
