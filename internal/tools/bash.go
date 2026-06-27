package tools

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

const (
	bashTimeoutSec = 120
	bashMaxOutput  = 64 * 1024
)

var bashSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"command": map[string]any{
			"type":        "string",
			"description": "The shell command to execute.",
		},
	},
	"required": []any{"command"},
}

// BashTool executes a shell command and returns its combined output.
type BashTool struct {
	BaseTool
	Cwd string
}

// NewBashTool constructs a BashTool that runs commands in cwd.
func NewBashTool(cwd string) *BashTool {
	return &BashTool{
		BaseTool: BaseTool{
			ToolDef: ai.Tool{
				Name:        "bash",
				Description: "Execute a shell command and return its combined stdout/stderr.",
				Parameters:  bashSchema,
			},
			ToolLabel:    "Bash",
			// Shell execution must be sequential to avoid races on the cwd.
			ToolExecMode: agent.ToolExecutionSequential,
		},
		Cwd: cwd,
	}
}

// Execute runs the bash tool.
func (t *BashTool) Execute(ctx context.Context, toolCallID string, params map[string]any, onUpdate agent.AgentToolUpdateCallback) (agent.AgentToolResult, error) {
	command, _ := params["command"].(string)
	if command == "" {
		return agent.AgentToolResult{}, fmt.Errorf("command is required")
	}

	// Use bash on POSIX, cmd fallback handled via shell selection. The agent
	// runs on the developer's machine; bash is expected (Git Bash on Windows).
	cctx, cancel := context.WithTimeout(ctx, bashTimeoutSec*time.Second)
	defer cancel()

	var cmd *exec.Cmd
	shell := "bash"
	if _, err := exec.LookPath(shell); err != nil {
		// Fall back to sh.
		shell = "sh"
	}
	cmd = exec.CommandContext(cctx, shell, "-c", command)
	if t.Cwd != "" {
		cmd.Dir = t.Cwd
	}

	var combined bytes.Buffer
	cmd.Stdout = &combined
	cmd.Stderr = &combined

	runErr := cmd.Run()
	output := combined.String()

	isError := false
	if runErr != nil {
		isError = true
		output = output + "\n[error: " + runErr.Error() + "]"
	}
	if len(output) > bashMaxOutput {
		output = output[:bashMaxOutput] + fmt.Sprintf("\n\n[... output truncated at %d bytes ...]", bashMaxOutput)
	}

	return agent.AgentToolResult{
		Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: strings.TrimSpace(output)}},
		Details: map[string]any{
			"command":  command,
			"is_error": isError,
		},
	}, nil
}
