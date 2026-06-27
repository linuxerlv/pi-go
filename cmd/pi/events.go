package main

import (
	"fmt"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

// printEvent renders an agent event to stdout/stderr. Assistant text is printed
// to stdout (streaming); lifecycle/tool metadata goes to stderr in dim ANSI.
func printEvent(ev agent.AgentEvent, verbose bool) {
	switch e := ev.(type) {
	case agent.AgentStartEvent:
		fmt.Fprintln(osStderr(), dim("[agent_start]"))
	case agent.AgentEndEvent:
		fmt.Fprintln(osStderr(), dim("[agent_end]"))
	case agent.TurnStartEvent:
		fmt.Fprintln(osStderr(), dim("[turn_start]"))
	case agent.TurnEndEvent:
		fmt.Fprintln(osStderr(), dim("[turn_end]"))

	case agent.MessageStartEvent:
		switch m := e.Message.(type) {
		case ai.UserMessage:
			// Don't echo the prompt back.
		case ai.AssistantMessage:
			// Streaming will emit deltas via MessageUpdateEvent.
		case ai.ToolResultMessage:
			fmt.Fprintf(osStderr(), dim("[tool_result %s isError=%v]\n"), m.ToolName, m.IsError)
		}
	case agent.MessageUpdateEvent:
		if d, ok := e.AssistantMessageEvent.(ai.TextDeltaEvent); ok {
			fmt.Print(d.Delta)
		} else if d, ok := e.AssistantMessageEvent.(ai.ThinkingDeltaEvent); ok {
			fmt.Fprint(osStderr(), dim(d.Delta))
		}
	case agent.MessageEndEvent:
		if m, ok := e.Message.(ai.AssistantMessage); ok {
			fmt.Fprintln(osStderr())
			if m.StopReason == ai.StopError || m.StopReason == ai.StopAborted {
				fmt.Fprintln(osStderr(), red(fmt.Sprintf("[%s] %s", m.StopReason, m.ErrorMessage)))
			}
		}

	case agent.ToolExecutionStartEvent:
		if verbose {
			fmt.Fprintf(osStderr(), dim("[tool_start %s] %s\n"), e.ToolName, ai.MustJSON(e.Args))
		} else {
			fmt.Fprintf(osStderr(), dim("[tool_start %s]\n"), e.ToolName)
		}
	case agent.ToolExecutionUpdateEvent:
		// Partial updates are noisy; only show in verbose.
		if verbose {
			fmt.Fprintf(osStderr(), dim("[tool_update %s]\n"), e.ToolName)
		}
	case agent.ToolExecutionEndEvent:
		if verbose {
			fmt.Fprintf(osStderr(), dim("[tool_end %s isError=%v] %s\n"), e.ToolName, e.IsError, toolResultPreview(e.Result))
		} else {
			fmt.Fprintf(osStderr(), dim("[tool_end %s isError=%v]\n"), e.ToolName, e.IsError)
		}
	}
}

func toolResultPreview(r agent.AgentToolResult) string {
	var out string
	for _, b := range r.Content {
		if t, ok := b.(ai.TextContent); ok {
			out += t.Text
		}
	}
	const max = 200
	if len(out) > max {
		out = out[:max] + "..."
	}
	out = stringsReplaceAll(out, "\n", " ")
	return out
}
