package tools

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

// defaultMaxBytes is the soft cap on bytes returned by the read tool.
const defaultMaxBytes = 256 * 1024

// readSchema is the JSON Schema for the read tool's parameters.
var readSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"path": map[string]any{
			"type":        "string",
			"description": "Path to the file to read (relative or absolute).",
		},
		"offset": map[string]any{
			"type":        "integer",
			"description": "Line number to start reading from (1-indexed).",
		},
		"limit": map[string]any{
			"type":        "integer",
			"description": "Maximum number of lines to read.",
		},
	},
	"required": []any{"path"},
}

// ReadTool reads a file from the local filesystem and returns its contents
// (with optional line range and a truncation note for large files).
type ReadTool struct {
	BaseTool
	Cwd string
}

// NewReadTool constructs a ReadTool anchored at cwd (for resolving relative
// paths).
func NewReadTool(cwd string) *ReadTool {
	return &ReadTool{
		BaseTool: BaseTool{
			ToolDef: ai.Tool{
				Name:        "read",
				Description: "Read the contents of a file. Supports an optional line offset and limit.",
				Parameters:  readSchema,
			},
			ToolLabel: "Read",
		},
		Cwd: cwd,
	}
}

// Execute runs the read tool.
func (t *ReadTool) Execute(ctx context.Context, toolCallID string, params map[string]any, onUpdate agent.AgentToolUpdateCallback) (agent.AgentToolResult, error) {
	path, _ := params["path"].(string)
	if path == "" {
		return agent.AgentToolResult{}, fmt.Errorf("path is required")
	}

	abs := path
	if !filepath.IsAbs(path) {
		abs = filepath.Join(t.Cwd, path)
	}

	info, err := os.Stat(abs)
	if err != nil {
		return agent.AgentToolResult{}, err
	}
	if info.IsDir() {
		return agent.AgentToolResult{}, fmt.Errorf("path is a directory: %s", path)
	}

	f, err := os.Open(abs)
	if err != nil {
		return agent.AgentToolResult{}, err
	}
	defer f.Close()

	offset := 1
	if v, ok := params["offset"]; ok {
		if n, err := toInt(v); err == nil && n > 0 {
			offset = n
		}
	}
	limit := 0
	if v, ok := params["limit"]; ok {
		if n, err := toInt(v); err == nil && n > 0 {
			limit = n
		}
	}

	scanner := bufio.NewScanner(f)
	// Allow large lines (up to 1MB) to avoid scanner token-too-long errors.
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var lines []string
	lineNo := 0
	bytesRead := 0
	truncated := false
	for scanner.Scan() {
		lineNo++
		if lineNo < offset {
			continue
		}
		line := scanner.Text()
		bytesRead += len(line) + 1
		if bytesRead > defaultMaxBytes {
			truncated = true
			break
		}
		lines = append(lines, fmt.Sprintf("%6d\t%s", lineNo, line))
		if limit > 0 && lineNo-offset+1 >= limit {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		return agent.AgentToolResult{}, err
	}

	var sb strings.Builder
	if offset > 1 || limit > 0 {
		fmt.Fprintf(&sb, "(showing lines %d-%d of %s)\n", offset, offset+len(lines)-1, strconv.Itoa(lineNo))
	}
	sb.WriteString(strings.Join(lines, "\n"))
	if truncated {
		fmt.Fprintf(&sb, "\n\n[... file truncated at %d bytes ...]", defaultMaxBytes)
	}

	return agent.AgentToolResult{
		Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: sb.String()}},
		Details: map[string]any{
			"path":      path,
			"truncated": truncated,
			"lines":     len(lines),
		},
	}, nil
}

func toInt(v any) (int, error) {
	switch n := v.(type) {
	case int:
		return n, nil
	case int64:
		return int(n), nil
	case float64:
		return int(n), nil
	case string:
		return strconv.Atoi(n)
	}
	return 0, fmt.Errorf("not an int: %T", v)
}
