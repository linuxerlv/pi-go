package tools

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

var editSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"path": map[string]any{
			"type":        "string",
			"description": "Path to the file to edit (relative or absolute).",
		},
		"edits": map[string]any{
			"type":        "array",
			"description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping edits.",
			"items": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"oldText": map[string]any{
						"type":        "string",
						"description": "Exact text for one targeted replacement. Must be unique in the file.",
					},
					"newText": map[string]any{
						"type":        "string",
						"description": "Replacement text for this targeted edit.",
					},
				},
				"required":             []any{"oldText", "newText"},
				"additionalProperties": false,
			},
		},
	},
	"required":             []any{"path", "edits"},
	"additionalProperties": false,
}

// EditTool applies targeted text replacements to a file. Each edit's oldText
// must match exactly once in the original file; all edits are applied against
// the original content (not incrementally).
type EditTool struct {
	BaseTool
}

// NewEditTool constructs an EditTool anchored at cwd.
func NewEditTool(cwd string) *EditTool {
	return &EditTool{
		BaseTool: BaseTool{
			ToolDef: ai.Tool{
				Name:        "edit",
				Description: "Apply targeted text replacements to a file. Each oldText must be unique in the file.",
				Parameters:  editSchema,
			},
			ToolLabel: "Edit",
			Cwd:       cwd,
		},
	}
}

// Execute runs the edit tool.
func (t *EditTool) Execute(ctx context.Context, toolCallID string, params map[string]any, onUpdate agent.AgentToolUpdateCallback) (agent.AgentToolResult, error) {
	path, err := pathParam(params)
	if err != nil {
		return agent.AgentToolResult{}, err
	}
	editsAny, _ := params["edits"].([]any)
	if len(editsAny) == 0 {
		return agent.AgentToolResult{}, fmt.Errorf("edits is required and must be non-empty")
	}

	type edit struct{ oldText, newText string }
	edits := make([]edit, 0, len(editsAny))
	for i, e := range editsAny {
		em, ok := e.(map[string]any)
		if !ok {
			return agent.AgentToolResult{}, fmt.Errorf("edits[%d] is not an object", i)
		}
		oldText, _ := em["oldText"].(string)
		newText, _ := em["newText"].(string)
		if oldText == "" {
			return agent.AgentToolResult{}, fmt.Errorf("edits[%d].oldText is required", i)
		}
		edits = append(edits, edit{oldText, newText})
	}

	abs := t.resolvePath(path)
	original, err := os.ReadFile(abs)
	if err != nil {
		return agent.AgentToolResult{}, err
	}
	content := string(original)

	// Validate all edits against the original content first (non-incremental).
	for i, e := range edits {
		count := strings.Count(content, e.oldText)
		if count == 0 {
			return agent.AgentToolResult{
				Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: fmt.Sprintf("Edit %d failed: oldText not found in %s", i, path)}},
				Details: map[string]any{"path": path, "failedEditIndex": i, "reason": "not found"},
			}, fmt.Errorf("edits[%d].oldText not found in file", i)
		}
		if count > 1 {
			return agent.AgentToolResult{
				Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: fmt.Sprintf("Edit %d failed: oldText matches %d times in %s (must be unique)", i, count, path)}},
				Details: map[string]any{"path": path, "failedEditIndex": i, "reason": "not unique", "matches": count},
			}, fmt.Errorf("edits[%d].oldText matches %d times (must be unique)", i, count)
		}
	}

	// Apply edits against the original content sequentially. Since each oldText
	// is unique, order does not matter and no overlap is possible.
	for _, e := range edits {
		content = strings.Replace(content, e.oldText, e.newText, 1)
	}

	if err := os.WriteFile(abs, []byte(content), 0o644); err != nil {
		return agent.AgentToolResult{}, err
	}

	return agent.AgentToolResult{
		Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: fmt.Sprintf("Applied %d edit(s) to %s", len(edits), path)}},
		Details: map[string]any{"path": path, "editsApplied": len(edits)},
	}, nil
}
