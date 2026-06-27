package tui

import (
	"strings"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

// AgentView is a differential-rendered view of an agent run. It accumulates
// assistant text, tool-call blocks, and a status line, and re-renders the frame
// on each update using Terminal.RenderDifferential so only changed rows are
// rewritten.
type AgentView struct {
	term     *Terminal
	title    string
	text     strings.Builder
	tools    []toolBlock
	status   string
	width    int
}

type toolBlock struct {
	name    string
	status  string // "running" | "ok" | "error"
	preview string
}

// NewAgentView constructs an AgentView on term with a title.
func NewAgentView(term *Terminal, title string) *AgentView {
	return &AgentView{term: term, title: title, width: term.Width(), status: "idle"}
}

// Start clears the screen and renders the initial frame.
func (v *AgentView) Start() {
	v.term.ClearScreen()
	v.term.HideCursor()
	v.render()
}

// Stop shows the cursor (call at run end).
func (v *AgentView) Stop() {
	v.term.ShowCursor()
}

// HandleEvent updates the view from an agent event and re-renders.
func (v *AgentView) HandleEvent(ev agent.AgentEvent) {
	switch e := ev.(type) {
	case agent.AgentStartEvent:
		v.status = "running"
	case agent.AgentEndEvent:
		v.status = "done"
	case agent.TurnStartEvent:
		v.status = "thinking"
	case agent.TurnEndEvent:
		v.status = "turn done"
	case agent.MessageUpdateEvent:
		if d, ok := e.AssistantMessageEvent.(ai.TextDeltaEvent); ok {
			v.text.WriteString(d.Delta)
			v.status = "streaming"
		}
	case agent.MessageEndEvent:
		if _, ok := e.Message.(ai.AssistantMessage); ok {
			v.text.WriteString("\n")
		}
	case agent.ToolExecutionStartEvent:
		v.tools = append(v.tools, toolBlock{name: e.ToolName, status: "running"})
		v.status = "tool: " + e.ToolName
	case agent.ToolExecutionEndEvent:
		for i := len(v.tools) - 1; i >= 0; i-- {
			if v.tools[i].name == e.ToolName && v.tools[i].status == "running" {
				if e.IsError {
					v.tools[i].status = "error"
				} else {
					v.tools[i].status = "ok"
				}
				v.tools[i].preview = toolResultPreview(e.Result)
				break
			}
		}
	}
	v.render()
}

func toolResultPreview(r agent.AgentToolResult) string {
	var out string
	for _, b := range r.Content {
		if t, ok := b.(ai.TextContent); ok {
			out += t.Text
		}
	}
	out = strings.ReplaceAll(out, "\n", " ")
	const max = 80
	if len(out) > max {
		out = out[:max] + "…"
	}
	return out
}

// render composes the frame and differentially renders it.
func (v *AgentView) render() {
	f := NewFrame()
	f.Append(v.title)
	f.Append(strings.Repeat("─", min(v.width, 60)))
	// Assistant text (wrapped naively by splitting on newlines).
	for _, line := range strings.Split(v.text.String(), "\n") {
		f.Append(line)
	}
	// Tool blocks.
	if len(v.tools) > 0 {
		f.Append("")
		f.Append("tools:")
		for _, tb := range v.tools {
			marker := "…"
			switch tb.status {
			case "ok":
				marker = "✓"
			case "error":
				marker = "✗"
			}
			f.Appendf("  %s %s", marker, tb.name)
			if tb.preview != "" {
				f.Appendf("      %s", tb.preview)
			}
		}
	}
	// Status bar.
	f.Append("")
	f.Append(strings.Repeat("─", min(v.width, 60)))
	f.Appendf("[%s]", v.status)
	v.term.RenderDifferential(f.Rows())
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
