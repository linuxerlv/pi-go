package tools

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestReadToolMissingFile verifies reading a non-existent file returns an error.
func TestReadToolMissingFile(t *testing.T) {
	r := NewReadTool(t.TempDir())
	_, err := r.Execute(context.Background(), "tc", map[string]any{"path": "nope.txt"}, nil)
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

// TestReadToolDirectoryReturnsError verifies reading a directory errors.
func TestReadToolDirectoryReturnsError(t *testing.T) {
	dir := t.TempDir()
	r := NewReadTool(dir)
	_, err := r.Execute(context.Background(), "tc", map[string]any{"path": "."}, nil)
	if err == nil {
		t.Fatal("expected error for directory read")
	}
}

// TestReadToolOffsetAndLimit verifies offset/limit slicing.
func TestReadToolOffsetAndLimit(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	os.WriteFile(path, []byte("l1\nl2\nl3\nl4\nl5\n"), 0o644)
	r := NewReadTool(dir)
	res, err := r.Execute(context.Background(), "tc", map[string]any{"path": "f.txt", "offset": 2, "limit": 2}, nil)
	if err != nil {
		t.Fatal(err)
	}
	txt := textOf(res.Content)
	if !strings.Contains(txt, "l2") || !strings.Contains(txt, "l3") {
		t.Fatalf("expected l2/l3 in range, got: %q", txt)
	}
	if strings.Contains(txt, "l1") || strings.Contains(txt, "l4") {
		t.Fatalf("range should exclude l1/l4, got: %q", txt)
	}
}

// TestReadToolResolvesRelativePath verifies relative paths resolve against Cwd.
func TestReadToolResolvesRelativePath(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "rel.txt"), []byte("relative"), 0o644)
	r := NewReadTool(dir)
	res, err := r.Execute(context.Background(), "tc", map[string]any{"path": "rel.txt"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(textOf(res.Content), "relative") {
		t.Fatalf("expected relative content, got %q", textOf(res.Content))
	}
}

// TestBashToolNonZeroExit verifies a failing command yields isError in details.
func TestBashToolNonZeroExit(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("bash tool tested on POSIX only")
	}
	b := NewBashTool(t.TempDir())
	res, err := b.Execute(context.Background(), "tc", map[string]any{"command": "exit 3"}, nil)
	if err != nil {
		t.Fatalf("Execute should not return error for non-zero exit: %v", err)
	}
	details, ok := res.Details.(map[string]any)
	if !ok {
		t.Fatalf("expected map details, got %T", res.Details)
	}
	if isErr, _ := details["is_error"].(bool); !isErr {
		t.Fatal("expected is_error=true for non-zero exit")
	}
}

// TestBashToolOutputTruncation verifies output beyond bashMaxOutput is truncated.
func TestBashToolOutputTruncation(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("bash tool tested on POSIX only")
	}
	b := NewBashTool(t.TempDir())
	// Print more than bashMaxOutput bytes.
	res, err := b.Execute(context.Background(), "tc", map[string]any{"command": "yes x | head -c 100000"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	txt := textOf(res.Content)
	if !strings.Contains(txt, "truncated") {
		t.Fatalf("expected truncation marker, got len=%d", len(txt))
	}
}

// TestWriteToolOverwritesExisting verifies writing to an existing file replaces
// its content.
func TestWriteToolOverwritesExisting(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	os.WriteFile(path, []byte("old"), 0o644)
	w := NewWriteTool(dir)
	if _, err := w.Execute(context.Background(), "tc", map[string]any{"path": "f.txt", "content": "new"}, nil); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(path)
	if string(b) != "new" {
		t.Fatalf("expected 'new', got %q", b)
	}
}

// TestGlobToolNoMatch verifies no matches yields an empty (non-error) result.
func TestGlobToolNoMatch(t *testing.T) {
	gl := NewGlobTool(t.TempDir())
	res, err := gl.Execute(context.Background(), "tc", map[string]any{"pattern": "*.nonexistent"}, nil)
	if err != nil {
		t.Fatalf("no-match glob should not error: %v", err)
	}
	// Result is non-nil content (an informational message), but must not list files.
	if strings.Contains(textOf(res.Content), ".nonexistent") {
		t.Fatalf("unexpected match in no-match glob: %q", textOf(res.Content))
	}
}

// TestEditToolMissingFile verifies editing a non-existent file errors.
func TestEditToolMissingFile(t *testing.T) {
	e := NewEditTool(t.TempDir())
	_, err := e.Execute(context.Background(), "tc", map[string]any{
		"path":  "nope.txt",
		"edits": []any{map[string]any{"oldText": "a", "newText": "b"}},
	}, nil)
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

// TestEditToolOldTextNotFound verifies a non-matching oldText errors.
func TestEditToolOldTextNotFound(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	os.WriteFile(path, []byte("alpha\n"), 0o644)
	e := NewEditTool(dir)
	_, err := e.Execute(context.Background(), "tc", map[string]any{
		"path":  "f.txt",
		"edits": []any{map[string]any{"oldText": "zzz", "newText": "b"}},
	}, nil)
	if err == nil {
		t.Fatal("expected error for oldText not found")
	}
}

// TestGrepToolNoMatch verifies grep with no matches does not error.
func TestGrepToolNoMatch(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "f.txt"), []byte("nothing here\n"), 0o644)
	g := NewGrepTool(dir)
	res, err := g.Execute(context.Background(), "tc", map[string]any{"pattern": "zzz"}, nil)
	if err != nil {
		t.Fatalf("no-match grep should not error: %v", err)
	}
	if strings.Contains(textOf(res.Content), "f.txt:") {
		t.Fatalf("unexpected match in no-match grep: %q", textOf(res.Content))
	}
}

// TestGrepToolBadRegex verifies an invalid regex errors.
func TestGrepToolBadRegex(t *testing.T) {
	g := NewGrepTool(t.TempDir())
	_, err := g.Execute(context.Background(), "tc", map[string]any{"pattern": "(unclosed"}, nil)
	if err == nil {
		t.Fatal("expected error for invalid regex")
	}
}
