package tools

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

const grepDefaultLimit = 100

var grepSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"pattern":    map[string]any{"type": "string", "description": "Search pattern (regex or literal string)."},
		"path":       map[string]any{"type": "string", "description": "Directory or file to search (default: current directory)."},
		"glob":       map[string]any{"type": "string", "description": "Filter files by glob pattern, e.g. '*.go'."},
		"ignoreCase": map[string]any{"type": "boolean", "description": "Case-insensitive search (default: false)."},
		"literal":    map[string]any{"type": "boolean", "description": "Treat pattern as literal string (default: false)."},
		"context":    map[string]any{"type": "integer", "description": "Lines of context before and after each match (default: 0)."},
		"limit":      map[string]any{"type": "integer", "description": "Maximum matches to return (default: 100)."},
	},
	"required": []any{"pattern"},
}

// GrepTool searches file contents for a pattern (regex or literal).
type GrepTool struct {
	BaseTool
	Cwd string
}

// NewGrepTool constructs a GrepTool anchored at cwd.
func NewGrepTool(cwd string) *GrepTool {
	return &GrepTool{
		BaseTool: BaseTool{
			ToolDef: ai.Tool{
				Name:        "grep",
				Description: "Search file contents for a pattern (regex or literal). Returns matching lines with file:line prefixes.",
				Parameters:  grepSchema,
			},
			ToolLabel: "Grep",
		},
		Cwd: cwd,
	}
}

// Execute runs the grep tool.
func (t *GrepTool) Execute(ctx context.Context, toolCallID string, params map[string]any, onUpdate agent.AgentToolUpdateCallback) (agent.AgentToolResult, error) {
	pattern, _ := params["pattern"].(string)
	if pattern == "" {
		return agent.AgentToolResult{}, fmt.Errorf("pattern is required")
	}
	searchPath, _ := params["path"].(string)
	if searchPath == "" {
		searchPath = "."
	}
	if !filepath.IsAbs(searchPath) {
		searchPath = filepath.Join(t.Cwd, searchPath)
	}
	globPattern, _ := params["glob"].(string)
	ignoreCase, _ := params["ignoreCase"].(bool)
	literal, _ := params["literal"].(bool)
	contextLines := 0
	if v, ok := params["context"]; ok {
		if n, err := toInt(v); err == nil && n > 0 {
			contextLines = n
		}
	}
	limit := grepDefaultLimit
	if v, ok := params["limit"]; ok {
		if n, err := toInt(v); err == nil && n > 0 {
			limit = n
		}
	}

	// Compile pattern.
	pat := pattern
	if literal {
		pat = regexp.QuoteMeta(pat)
	}
	flags := ""
	if ignoreCase {
		flags = "(?i)"
	}
	re, err := regexp.Compile(flags + pat)
	if err != nil {
		return agent.AgentToolResult{}, fmt.Errorf("invalid pattern: %w", err)
	}

	var files []string
	info, err := os.Stat(searchPath)
	if err != nil {
		return agent.AgentToolResult{}, err
	}
	if info.IsDir() {
		err = filepath.WalkDir(searchPath, func(p string, d os.DirEntry, walkErr error) error {
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
			if globPattern != "" {
				matched, _ := filepath.Match(globPattern, d.Name())
				if !matched {
					// Also try matching against the path relative to the search root.
					rel, _ := filepath.Rel(searchPath, p)
					if matched2, _ := filepath.Match(globPattern, rel); !matched2 {
						return nil
					}
				}
			}
			files = append(files, p)
			return nil
		})
		if err != nil {
			return agent.AgentToolResult{}, err
		}
	} else {
		files = []string{searchPath}
	}

	var results []string
	matches := 0
	limitReached := false
	for _, file := range files {
		if matches >= limit {
			limitReached = true
			break
		}
		f, err := os.Open(file)
		if err != nil {
			continue
		}
		var lines []string
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			lines = append(lines, scanner.Text())
		}
		f.Close()
		if err := scanner.Err(); err != nil {
			continue
		}

		rel, _ := filepath.Rel(t.Cwd, file)
		if rel == "" {
			rel = file
		}
		for i, line := range lines {
			if re.MatchString(line) {
				if matches >= limit {
					limitReached = true
					break
				}
				if contextLines > 0 {
					start := i - contextLines
					if start < 0 {
						start = 0
					}
					end := i + contextLines
					if end >= len(lines) {
						end = len(lines) - 1
					}
					for j := start; j <= end; j++ {
						marker := "  "
						if j == i {
							marker = "> "
						}
						results = append(results, fmt.Sprintf("%s%s:%d: %s", marker, rel, j+1, lines[j]))
					}
					results = append(results, "")
				} else {
					results = append(results, fmt.Sprintf("%s:%d: %s", rel, i+1, line))
				}
				matches++
			}
		}
	}

	out := strings.Join(results, "\n")
	if limitReached {
		out += fmt.Sprintf("\n\n[... match limit (%d) reached ...]", limit)
	}
	if matches == 0 {
		out = "No matches found."
	}
	return agent.AgentToolResult{
		Content: []ai.ContentBlock{ai.TextContent{Type: "text", Text: out}},
		Details: map[string]any{"matches": matches, "limitReached": limitReached},
	}, nil
}
