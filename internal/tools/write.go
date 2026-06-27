package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

var writeSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"path": map[string]any{
			"type":        "string",
			"description": "Path to the file to write (relative or absolute).",
		},
		"content": map[string]any{
			"type":        "string",
			"description": "Content to write to the file. Overwrites existing content.",
		},
	},
	"required":             []any{"path", "content"},
	"additionalProperties": false,
}

// WriteTool writes content to a file, creating parent directories as needed.
type WriteTool struct {
	BaseTool
	Cwd string
}

// NewWriteTool constructs a WriteTool anchored at cwd.
func NewWriteTool(cwd string) *WriteTool {
	return &WriteTool{
		BaseTool: BaseTool{
			ToolDef: ai.Tool{
				Name:        "write",
				Description: "Write content to a file (overwrites). Creates parent directories.",
				Parameters:  writeSchema,
			},
			ToolLabel: "Write",
		},
		Cwd: cwd,
	}
}

// Execute runs the write tool.
func (t *WriteTool) Execute(ctx context.Context, toolCallID string, params map[string]any, onUpdate agent.AgentToolUpdateCallback) (agent.AgentToolResult, error) {
	path, _ := params["path"].(string)
	content, _ := params["content"].(string)
	if path == "" {
		return agent.AgentToolResult{}, fmt.Errorf("path is required")
	}

	abs := path
	if !filepath.IsAbs(path) {
		abs = filepath.Join(t.Cwd, path)
	}
	dir := filepath.Dir(abs)
	if dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return agent.AgentToolResult{}, err
		}
	}
	if err := os.WriteFile(abs, []byte(content), 0o644); err != nil {
		return agent.AgentToolResult{}, err
	}

	written := len(content)
	return agent.AgentToolResult{
		Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: fmt.Sprintf("Wrote %d bytes to %s", written, path)}},
		Details: map[string]any{"path": path, "bytes": written},
	}, nil
}
