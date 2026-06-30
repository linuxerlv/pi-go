package main

import (
	"io"
	"os"
	"strings"
	"testing"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

func TestToolResultPreview(t *testing.T) {
	r := agent.AgentToolResult{Content: []ai.ContentBlock{
		ai.TextContent{Type: "text", Text: "line1\n"},
		ai.TextContent{Type: "text", Text: "line2"},
	}}
	got := toolResultPreview(r)
	if strings.Contains(got, "\n") {
		t.Fatalf("newlines should be flattened: %q", got)
	}
	if !strings.Contains(got, "line1") || !strings.Contains(got, "line2") {
		t.Fatalf("expected both lines, got %q", got)
	}

	long := strings.Repeat("x", 300)
	got2 := toolResultPreview(agent.AgentToolResult{Content: []ai.ContentBlock{
		ai.TextContent{Type: "text", Text: long},
	}})
	if !strings.HasSuffix(got2, "...") {
		t.Fatalf("expected truncation suffix, got len=%d", len(got2))
	}
	if len(got2) > 203 {
		t.Fatalf("truncated preview too long: %d", len(got2))
	}

	got3 := toolResultPreview(agent.AgentToolResult{Content: []ai.ContentBlock{
		ai.ImageContent{Type: "image", Data: "x"},
	}})
	if got3 != "" {
		t.Fatalf("expected empty preview for non-text, got %q", got3)
	}
}

// printEvent routes assistant text deltas to stdout and lifecycle/tool metadata
// to stderr. We capture both via pipes (synchronous io.ReadAll after closing
// the write end) to verify routing without asserting exact ANSI bytes.
func TestPrintEventRouting(t *testing.T) {
	rOut, wOut, _ := os.Pipe()
	oldStdout := os.Stdout
	os.Stdout = wOut

	rErr, wErr, _ := os.Pipe()
	oldStderr := os.Stderr
	os.Stderr = wErr

	printEvent(agent.MessageUpdateEvent{
		AssistantMessageEvent: ai.TextDeltaEvent{ContentIndex: 0, Delta: "hello"},
	}, false)
	printEvent(agent.ToolExecutionEndEvent{ToolName: "read", IsError: false}, false)
	printEvent(agent.AgentStartEvent{}, false)

	wOut.Close()
	os.Stdout = oldStdout
	wErr.Close()
	os.Stderr = oldStderr

	stdoutBytes, _ := io.ReadAll(rOut)
	rOut.Close()
	stderrBytes, _ := io.ReadAll(rErr)
	rErr.Close()

	if !strings.Contains(string(stdoutBytes), "hello") {
		t.Fatalf("expected text delta on stdout, got %q", stdoutBytes)
	}
	if !strings.Contains(string(stderrBytes), "tool_end read") {
		t.Fatalf("expected tool_end on stderr, got %q", stderrBytes)
	}
	if !strings.Contains(string(stderrBytes), "agent_start") {
		t.Fatalf("expected agent_start on stderr, got %q", stderrBytes)
	}
}

func TestPrintEventToolEndVerboseIncludesPreview(t *testing.T) {
	rErr, wErr, _ := os.Pipe()
	oldStderr := os.Stderr
	os.Stderr = wErr

	printEvent(agent.ToolExecutionEndEvent{
		ToolName: "read",
		IsError:  false,
		Result: agent.AgentToolResult{Content: []ai.ContentBlock{
			ai.TextContent{Type: "text", Text: "preview-text"},
		}},
	}, true)

	wErr.Close()
	os.Stderr = oldStderr
	buf, _ := io.ReadAll(rErr)
	rErr.Close()
	if !strings.Contains(string(buf), "preview-text") {
		t.Fatalf("verbose tool_end should include preview, got %q", buf)
	}
}

func TestPrintEventMessageEndError(t *testing.T) {
	rErr, wErr, _ := os.Pipe()
	oldStderr := os.Stderr
	os.Stderr = wErr

	printEvent(agent.MessageEndEvent{
		Message: ai.AssistantMessage{StopReason: ai.StopError, ErrorMessage: "boom"},
	}, false)

	wErr.Close()
	os.Stderr = oldStderr
	buf, _ := io.ReadAll(rErr)
	rErr.Close()
	if !strings.Contains(string(buf), "boom") {
		t.Fatalf("expected error message on stderr, got %q", buf)
	}
}
