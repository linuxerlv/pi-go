package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

// runMockServerMode is a configurable mock MCP server selected by env var:
//   MCP_TEST_MODE=silent    -> answers initialize then never responds to calls
//   MCP_TEST_MODE=rpcerror  -> returns a JSON-RPC error for tools/call
//   MCP_TEST_MODE=toolok    -> returns a tool result (for MCPTool adapter test)
func runMockServerMode() {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	enc := json.NewEncoder(os.Stdout)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var req struct {
			ID     *int64 `json:"id"`
			Method string `json:"method"`
		}
		if json.Unmarshal([]byte(line), &req) != nil {
			continue
		}
		if req.ID == nil {
			continue // notification
		}
		mode := os.Getenv("MCP_TEST_MODE")
		if req.Method == "initialize" {
			enc.Encode(map[string]any{
				"jsonrpc": "2.0", "id": *req.ID,
				"result": map[string]any{
					"protocolVersion": "2024-11-05",
					"capabilities":    map[string]any{},
					"serverInfo":      map[string]any{"name": "mock-mcp", "version": "0.1"},
				},
			})
			continue
		}
		switch mode {
		case "silent":
			// Never respond to tools/call -> caller's ctx will time out.
			continue
		case "rpcerror":
			enc.Encode(map[string]any{
				"jsonrpc": "2.0", "id": *req.ID,
				"error": map[string]any{"code": -32603, "message": "boom"},
			})
		case "toolok":
			enc.Encode(map[string]any{
				"jsonrpc": "2.0", "id": *req.ID,
				"result": map[string]any{
					"content": []map[string]any{{"type": "text", "text": "tool-result"}},
					"isError": false,
				},
			})
		default:
			enc.Encode(map[string]any{
				"jsonrpc": "2.0", "id": *req.ID,
				"error":   map[string]any{"code": -32601, "message": "unknown mode"},
			})
		}
	}
}

// startModeServer launches the test binary in the given mock mode and returns
// the client (caller defers Close).
func startModeServer(t *testing.T, mode string) *Client {
	t.Helper()
	if os.Getenv("MCP_TEST_MODE_CHILD") == "1" {
		runMockServerMode()
		os.Exit(0)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	client, err := Start(ctx, exe, nil, []string{"MCP_TEST_MODE_CHILD=1", "MCP_TEST_MODE=" + mode})
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	return client
}

// TestCloseIsIdempotent verifies multiple Close calls do not panic.
func TestCloseIsIdempotent(t *testing.T) {
	client := startModeServer(t, "toolok")
	if err := client.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := client.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if err := client.Close(); err != nil {
		t.Fatalf("third Close: %v", err)
	}
}

// TestCallCancelledByContext verifies that a call whose server never responds
// returns ctx.Err() when the context expires (reader goroutine does not hang).
func TestCallCancelledByContext(t *testing.T) {
	client := startModeServer(t, "silent")
	defer client.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	_, err := client.CallTool(ctx, "echo", map[string]any{"text": "x"})
	if err == nil {
		t.Fatal("expected error on ctx cancellation, got nil")
	}
}

// TestCallRPCError verifies a JSON-RPC error response surfaces as a call error.
func TestCallRPCError(t *testing.T) {
	client := startModeServer(t, "rpcerror")
	defer client.Close()
	_, err := client.CallTool(context.Background(), "echo", map[string]any{"text": "x"})
	if err == nil {
		t.Fatal("expected error for RPC error response, got nil")
	}
	if !strings.Contains(err.Error(), "error") {
		t.Fatalf("expected error mentioning 'error', got %v", err)
	}
}

// TestMCPToolAdapter verifies MCPTool.Execute forwards to the server and maps
// the text content into an AgentToolResult.
func TestMCPToolAdapter(t *testing.T) {
	client := startModeServer(t, "toolok")
	defer client.Close()
	tool := NewMCPTool(client, ToolDef{
		Name:        "echo",
		Description: "echo",
		InputSchema: map[string]any{"type": "object"},
	})
	// Def/Label/ExecutionMode smoke checks.
	if tool.Def().Name != "echo" {
		t.Fatalf("tool name = %q", tool.Def().Name)
	}
	if tool.Label() == "" {
		t.Fatal("label should be non-empty")
	}
	if tool.ExecutionMode() != "" {
		t.Fatalf("ExecutionMode should be empty, got %q", tool.ExecutionMode())
	}
	// PrepareArguments pass-through.
	args, err := tool.PrepareArguments(map[string]any{"text": "x"})
	if err != nil || args["text"] != "x" {
		t.Fatalf("PrepareArguments pass-through failed: %v %v", args, err)
	}
	res, err := tool.Execute(context.Background(), "tc1", map[string]any{"text": "x"}, nil)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if len(res.Content) == 0 {
		t.Fatal("expected non-empty content")
	}
	if len(res.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(res.Content))
	}
	// Details should carry the isError flag and source.
	details, ok := res.Details.(map[string]any)
	if !ok {
		t.Fatalf("expected map details, got %T", res.Details)
	}
	if details["source"] != "mcp:echo" {
		t.Fatalf("expected source mcp:echo, got %v", details["source"])
	}
}

// TestNotifyOnFullChannelDoesNotBlock verifies notify with a full stdin channel
// does not block (non-blocking send). We fill the channel then notify; the call
// must return promptly. This is a best-effort test of the non-blocking notify.
func TestNotifyOnFullChannelDoesNotBlock(t *testing.T) {
	client := startModeServer(t, "toolok")
	defer client.Close()
	// Fill the stdin channel (capacity 64) with notifications.
	for i := 0; i < 70; i++ {
		client.notify("notifications/test", map[string]any{"i": i})
	}
	// If notify were blocking, this would hang. The fact that we reach here
	// means non-blocking notify worked.
}
