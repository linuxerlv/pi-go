package tools

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/linuxerlv/pi-go/internal/ai"
)

func TestWriteToolCreatesFileAndDirs(t *testing.T) {
	dir := t.TempDir()
	w := NewWriteTool(dir)
	res, err := w.Execute(context.Background(), "tc", map[string]any{
		"path":    "sub/dir/file.txt",
		"content": "hello",
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "sub", "dir", "file.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != "hello" {
		t.Fatalf("expected 'hello', got %q", b)
	}
	if !strings.Contains(textOf(res.Content), "Wrote 5 bytes") {
		t.Fatalf("unexpected result: %v", res.Content)
	}
}

func TestEditToolAppliesUniqueReplacements(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	os.WriteFile(path, []byte("alpha\nbeta\ngamma\n"), 0o644)

	e := NewEditTool(dir)
	_, err := e.Execute(context.Background(), "tc", map[string]any{
		"path": "f.txt",
		"edits": []any{
			map[string]any{"oldText": "beta", "newText": "BETA"},
			map[string]any{"oldText": "alpha", "newText": "ALPHA"},
		},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(path)
	if string(b) != "ALPHA\nBETA\ngamma\n" {
		t.Fatalf("unexpected content: %q", b)
	}
}

func TestEditToolRejectsNonUnique(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	os.WriteFile(path, []byte("dup\ndup\n"), 0o644)

	e := NewEditTool(dir)
	_, err := e.Execute(context.Background(), "tc", map[string]any{
		"path":  "f.txt",
		"edits": []any{map[string]any{"oldText": "dup", "newText": "x"}},
	}, nil)
	if err == nil {
		t.Fatal("expected error for non-unique oldText")
	}
	if !strings.Contains(err.Error(), "matches 2 times") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestGrepToolFindsMatches(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.go"), []byte("package foo\nfunc Bar() {}\n"), 0o644)
	os.WriteFile(filepath.Join(dir, "b.txt"), []byte("no match here\n"), 0o644)

	g := NewGrepTool(dir)
	res, err := g.Execute(context.Background(), "tc", map[string]any{
		"pattern": "func",
		"glob":    "*.go",
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	out := textOf(res.Content)
	if !strings.Contains(out, "a.go:2:") || !strings.Contains(out, "Bar") {
		t.Fatalf("expected match in a.go, got: %q", out)
	}
	if strings.Contains(out, "b.txt") {
		t.Fatalf("b.txt should not match: %q", out)
	}
}

func TestGrepToolLiteralAndIgnoreCase(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "f.txt"), []byte("Hello World\nHELLO again\n"), 0o644)

	g := NewGrepTool(dir)
	res, err := g.Execute(context.Background(), "tc", map[string]any{
		"pattern":    "hello",
		"ignoreCase": true,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	out := textOf(res.Content)
	// Both lines should match (case-insensitive).
	if strings.Count(out, "f.txt:") != 2 {
		t.Fatalf("expected 2 matches, got: %q", out)
	}
}

func TestGlobToolDoubleStar(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "a", "b"), 0o755)
	os.WriteFile(filepath.Join(dir, "a", "b", "x.go"), []byte("x"), 0o644)
	os.WriteFile(filepath.Join(dir, "a", "y.go"), []byte("y"), 0o644)
	os.WriteFile(filepath.Join(dir, "z.txt"), []byte("z"), 0o644)

	gl := NewGlobTool(dir)
	res, err := gl.Execute(context.Background(), "tc", map[string]any{
		"pattern": "**/*.go",
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	out := textOf(res.Content)
	if !strings.Contains(out, "a/b/x.go") || !strings.Contains(out, "a/y.go") {
		t.Fatalf("expected both .go files, got: %q", out)
	}
	if strings.Contains(out, "z.txt") {
		t.Fatalf("z.txt should not match: %q", out)
	}
}

func TestGlobToolNoDoubleStar(t *testing.T) {
	if runtime.GOOS == "windows" {
		// filepath.Match behavior on backslash; we use ToSlash in tool. Skip on win.
	}
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.go"), []byte("a"), 0o644)
	os.WriteFile(filepath.Join(dir, "b.txt"), []byte("b"), 0o644)

	gl := NewGlobTool(dir)
	res, err := gl.Execute(context.Background(), "tc", map[string]any{
		"pattern": "*.go",
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	out := textOf(res.Content)
	if !strings.Contains(out, "a.go") || strings.Contains(out, "b.txt") {
		t.Fatalf("expected only a.go, got: %q", out)
	}
}

// silence unused import guards if ai is referenced only via textOf helper path.
var _ = ai.TextContent{}
