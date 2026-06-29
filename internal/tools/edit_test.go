package tools

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// writeFile is a small helper for edit tests.
func writeFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func readFile(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// editParams builds the validated params map the edit tool expects.
func editParams(path string, edits ...[2]string) map[string]any {
	arr := make([]any, len(edits))
	for i, e := range edits {
		arr[i] = map[string]any{"oldText": e[0], "newText": e[1]}
	}
	return map[string]any{"path": path, "edits": arr}
}

// TestEditAppliesMultipleNonOverlapping is the no-regression case: two
// independent edits both apply against the original.
func TestEditAppliesMultipleNonOverlapping(t *testing.T) {
	dir := t.TempDir()
	p := writeFile(t, dir, "f.txt", "alpha beta gamma")
	e := NewEditTool(dir)
	if _, err := e.Execute(context.Background(), "t", editParams(p,
		[2]string{"alpha", "A"},
		[2]string{"gamma", "G"},
	), nil); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if got := readFile(t, p); got != "A beta G" {
		t.Fatalf("got %q want %q", got, "A beta G")
	}
}

// TestEditNewTextContainingNextOldText is the core bug: A's newText contains
// B's oldText AND that copy lands BEFORE B's original occurrence. Sequential
// Replace (A then B) would replace the copy inside A's newText instead of B's
// original; positional apply matches the original unambiguously.
func TestEditNewTextContainingNextOldText(t *testing.T) {
	dir := t.TempDir()
	// Original: "foo" at pos 0, "baz" at pos 6. Both unique.
	p := writeFile(t, dir, "f.txt", "foo . baz")
	e := NewEditTool(dir)
	// Edit A: foo -> "baz" (newText equals B's oldText, placed at pos 0).
	// Edit B: baz -> "QUX".
	if _, err := e.Execute(context.Background(), "t", editParams(p,
		[2]string{"foo", "baz"},
		[2]string{"baz", "QUX"},
	), nil); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	// Positional: original foo(pos0)->"baz", original baz(pos6)->"QUX" => "baz . QUX".
	// Buggy sequential (A then B): "baz . baz" then replace first "baz" => "QUX . baz".
	if got := readFile(t, p); got != "baz . QUX" {
		t.Fatalf("positional apply mismatch: got %q want %q", got, "baz . QUX")
	}
}

// TestEditRejectsOverlappingSpans: two edits whose oldText ranges overlap in
// the original must be rejected.
func TestEditRejectsOverlappingSpans(t *testing.T) {
	dir := t.TempDir()
	// "aaaa": edit1 oldText "aa" (unique? Count("aaaa","aa")==3, not unique ->
	// rejected at validate). Use distinct overlap instead.
	p := writeFile(t, dir, "f.txt", "abcdef")
	e := NewEditTool(dir)
	// "abcd" and "cdef" overlap on "cd". Each is unique in "abcdef".
	_, err := e.Execute(context.Background(), "t", editParams(p,
		[2]string{"abcd", "X"},
		[2]string{"cdef", "Y"},
	), nil)
	if err == nil {
		t.Fatal("expected overlap error, got nil")
	}
}

// TestEditAdjacentSpansAllowed: adjacent (non-overlapping) edits apply.
func TestEditAdjacentSpansAllowed(t *testing.T) {
	dir := t.TempDir()
	p := writeFile(t, dir, "f.txt", "abcdef")
	e := NewEditTool(dir)
	// "abc" and "def" are adjacent (end==start), no overlap.
	if _, err := e.Execute(context.Background(), "t", editParams(p,
		[2]string{"abc", "X"},
		[2]string{"def", "Y"},
	), nil); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if got := readFile(t, p); got != "XY" {
		t.Fatalf("got %q want %q", got, "XY")
	}
}

// TestEditRejectsDuplicateOldText: the same oldText twice (two edits) now
// resolves to the same span -> overlap -> rejected.
func TestEditRejectsDuplicateOldText(t *testing.T) {
	dir := t.TempDir()
	p := writeFile(t, dir, "f.txt", "unique token here")
	e := NewEditTool(dir)
	_, err := e.Execute(context.Background(), "t", editParams(p,
		[2]string{"token", "A"},
		[2]string{"token", "B"},
	), nil)
	if err == nil {
		t.Fatal("expected duplicate-span overlap error, got nil")
	}
}
