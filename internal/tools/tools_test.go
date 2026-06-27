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

func TestReadToolReadsFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sample.txt")
	if err := os.WriteFile(path, []byte("hello\nworld\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	r := NewReadTool(dir)
	res, err := r.Execute(context.Background(), "tc1", map[string]any{"path": "sample.txt"}, nil)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	txt := textOf(res.Content)
	if !strings.Contains(txt, "hello") || !strings.Contains(txt, "world") {
		t.Fatalf("unexpected content: %q", txt)
	}
}

func TestBashToolRunsCommand(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("bash tool tested on POSIX only in this unit test")
	}
	dir := t.TempDir()
	b := NewBashTool(dir)
	res, err := b.Execute(context.Background(), "tc1", map[string]any{"command": "echo hi && pwd"}, nil)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	txt := textOf(res.Content)
	if !strings.Contains(txt, "hi") {
		t.Fatalf("expected output to contain 'hi', got: %q", txt)
	}
}

func textOf(blocks []ai.ContentBlock) string {
	var sb strings.Builder
	for _, b := range blocks {
		if t, ok := b.(ai.TextContent); ok {
			sb.WriteString(t.Text)
		}
	}
	return sb.String()
}
