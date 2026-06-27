// Package tools provides built-in agent tools (read, bash, edit, write, grep,
// glob) for the pi-go coding agent, plus a BaseTool helper for implementing
// custom tools.
package tools

import (
	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

// BaseTool provides the default AgentTool bookkeeping (definition, label,
// execution mode). Embed it in a concrete tool struct and implement Execute.
type BaseTool struct {
	ToolDef       ai.Tool
	ToolLabel     string
	ToolExecMode  agent.ToolExecutionMode
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
