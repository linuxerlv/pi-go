package mcp

import (
	"context"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

// MCPTool wraps an MCP tool definition as an agent.AgentTool. Each call is
// forwarded to the MCP server via tools/call.
type MCPTool struct {
	client *Client
	def    ToolDef
	label  string
}

// NewMCPTool wraps a single MCP tool as an AgentTool.
func NewMCPTool(client *Client, def ToolDef) *MCPTool {
	return &MCPTool{client: client, def: def, label: "MCP: " + def.Name}
}

// Def returns the tool definition (name/description/parameters).
func (t *MCPTool) Def() ai.Tool {
	return ai.Tool{
		Name:        t.def.Name,
		Description: t.def.Description,
		Parameters:  t.def.InputSchema,
	}
}

// Label returns a human-readable label.
func (t *MCPTool) Label() string { return t.label }

// PrepareArguments is a no-op pass-through (validation happens server-side).
func (t *MCPTool) PrepareArguments(args map[string]any) (map[string]any, error) {
	return args, nil
}

// ExecutionMode returns "" (use the loop default).
func (t *MCPTool) ExecutionMode() agent.ToolExecutionMode { return "" }

// Execute forwards the call to the MCP server and maps the result content into
// an AgentToolResult.
func (t *MCPTool) Execute(ctx context.Context, toolCallID string, params map[string]any, onUpdate agent.AgentToolUpdateCallback) (agent.AgentToolResult, error) {
	res, err := t.client.CallTool(ctx, t.def.Name, params)
	if err != nil {
		return agent.AgentToolResult{}, err
	}
	var blocks []ai.ContentBlock
	for _, c := range res.Content {
		if c.Type == "text" || c.Type == "" {
			blocks = append(blocks, ai.TextContent{Type: "text", Text: c.Text})
		}
	}
	if len(blocks) == 0 {
		blocks = []ai.ContentBlock{ai.TextContent{Type: "text", Text: ""}}
	}
	return agent.AgentToolResult{
		Content: blocks,
		Details: map[string]any{
			"isError": res.IsError,
			"source":  "mcp:" + t.def.Name,
		},
	}, nil
}
