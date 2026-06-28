package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

const globDefaultLimit = 100

var globSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"pattern": map[string]any{
			"type":        "string",
			"description": "Glob pattern, e.g. '**/*.go' or 'cmd/*.go'. Relative to cwd.",
		},
		"path": map[string]any{
			"type":        "string",
			"description": "Directory to search in (default: current directory).",
		},
		"limit": map[string]any{
			"type":        "integer",
			"description": "Maximum number of paths to return (default: 100).",
		},
	},
	"required": []any{"pattern"},
}

// GlobTool lists files matching a glob pattern.
type GlobTool struct {
	BaseTool
}

// NewGlobTool constructs a GlobTool anchored at cwd.
func NewGlobTool(cwd string) *GlobTool {
	return &GlobTool{
		BaseTool: BaseTool{
			ToolDef: ai.Tool{
				Name:        "glob",
				Description: "List files matching a glob pattern (e.g. '**/*.go'). Returns paths relative to cwd.",
				Parameters:  globSchema,
			},
			ToolLabel: "Glob",
			Cwd:       cwd,
		},
	}
}

// Execute runs the glob tool.
func (t *GlobTool) Execute(ctx context.Context, toolCallID string, params map[string]any, onUpdate agent.AgentToolUpdateCallback) (agent.AgentToolResult, error) {
	pattern, _ := params["pattern"].(string)
	if pattern == "" {
		return agent.AgentToolResult{}, fmt.Errorf("pattern is required")
	}
	searchDir, _ := params["path"].(string)
	if searchDir == "" {
		searchDir = t.Cwd
	}
	searchDir = t.resolvePath(searchDir)
	limit := globDefaultLimit
	if v, ok := params["limit"]; ok {
		if n, err := toInt(v); err == nil && n > 0 {
			limit = n
		}
	}

	var matches []string
	// Support ** glob by walking and matching the relative path.
	err := filepath.WalkDir(searchDir, func(p string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if d.IsDir() {
			base := d.Name()
			if base == ".git" || base == "node_modules" || base == ".pi-go" {
				return filepath.SkipDir
			}
			return nil
		}
		rel, err := filepath.Rel(searchDir, p)
		if err != nil {
			return nil
		}
		rel = filepath.ToSlash(rel)
		if globMatch(pattern, rel) {
			matches = append(matches, rel)
		}
		return nil
	})
	if err != nil {
		return agent.AgentToolResult{}, err
	}

	sort.Strings(matches)
	limitReached := false
	if len(matches) > limit {
		matches = matches[:limit]
		limitReached = true
	}

	out := strings.Join(matches, "\n")
	if len(matches) == 0 {
		out = "No files matched."
	}
	if limitReached {
		out += fmt.Sprintf("\n\n[... result limit (%d) reached ...]", limit)
	}
	return agent.AgentToolResult{
		Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: out}},
		Details: map[string]any{"count": len(matches), "limitReached": limitReached},
	}, nil
}

// globMatch matches a pattern with ** support against a slash-separated path.
// ** matches any number of path segments (including zero).
func globMatch(pattern, path string) bool {
	// Fast path: plain filepath.Match works for patterns without **.
	if !strings.Contains(pattern, "**") {
		ok, _ := filepath.Match(pattern, path)
		return ok
	}
	re, err := regexp.Compile(globToRegexp(pattern))
	if err != nil {
		return false
	}
	return re.MatchString(path)
}

// globToRegexp converts a glob pattern (with ** support) to a regexp string.
func globToRegexp(pattern string) string {
	var sb strings.Builder
	sb.WriteString("^")
	i := 0
	for i < len(pattern) {
		if i+1 < len(pattern) && pattern[i] == '*' && pattern[i+1] == '*' {
			// Optionally consume a following slash so **/ matches zero dirs.
			if i+2 < len(pattern) && pattern[i+2] == '/' {
				sb.WriteString("(?:.*/)?")
				i += 3
				continue
			}
			sb.WriteString(".*")
			i += 2
			continue
		}
		switch pattern[i] {
		case '*':
			sb.WriteString("[^/]*")
		case '?':
			sb.WriteString("[^/]")
		case '.', '(', ')', '+', '|', '^', '$', '\\', '{', '}':
			sb.WriteByte('\\')
			sb.WriteByte(pattern[i])
		default:
			sb.WriteByte(pattern[i])
		}
		i++
	}
	sb.WriteString("$")
	return sb.String()
}
