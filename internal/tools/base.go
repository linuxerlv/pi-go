// Package tools provides built-in agent tools (read, bash, edit, write, grep,
// glob) for the pi-go coding agent, plus a BaseTool helper for implementing
// custom tools.
package tools

import (
	"fmt"
	"path/filepath"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

// BaseTool provides the default AgentTool bookkeeping (definition, label,
// execution mode) plus shared helpers for path-based tools. Embed it in a
// concrete tool struct and implement Execute. Set Cwd so resolvePath can
// resolve relative paths.
type BaseTool struct {
	ToolDef      ai.Tool
	ToolLabel    string
	ToolExecMode agent.ToolExecutionMode
	Cwd          string
}

// Def returns the tool definition.
func (b BaseTool) Def() ai.Tool { return b.ToolDef }

// Label returns the human-readable label.
func (b BaseTool) Label() string {
	if b.ToolLabel != "" {
		return b.ToolLabel
	}
	return b.ToolDef.Name
}

// ExecutionMode returns the per-tool execution mode override ("" = default).
func (b BaseTool) ExecutionMode() agent.ToolExecutionMode { return b.ToolExecMode }

// pathParam extracts the "path" string from validated tool params, returning an
// error if it is missing. Shared by read/write/edit/grep/glob.
func pathParam(params map[string]any) (string, error) {
	path, _ := params["path"].(string)
	if path == "" {
		return "", fmt.Errorf("path is required")
	}
	return path, nil
}

// resolvePath resolves a (possibly relative) path against the tool's Cwd,
// returning an absolute path. Shared by all file tools to eliminate the
// repeated IsAbs/Join boilerplate.
func (b BaseTool) resolvePath(path string) string {
	if filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(b.Cwd, path)
}

// textResult builds a successful AgentToolResult from a single text block.
// Shared by tools whose result is primarily text.
func textResult(text string) agent.AgentToolResult {
	return agent.AgentToolResult{
		Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: text}},
	}
}

// textResultWithDetails is like textResult but attaches structured details.
func textResultWithDetails(text string, details any) agent.AgentToolResult {
	return agent.AgentToolResult{
		Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: text}},
		Details: details,
	}
}
