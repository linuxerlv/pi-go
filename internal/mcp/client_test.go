package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// This test uses the test-binary subprocess pattern: when MCP_TEST_SERVER=1,
// the process acts as a minimal MCP server (stdio JSON-RPC) and exits; the
// parent test launches it as the server under test.

func TestMCPClientListAndCall(t *testing.T) {
	if os.Getenv("MCP_TEST_SERVER") == "1" {
		runMockServer()
		return
	}

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	client, err := Start(ctx, exe, nil, []string{"MCP_TEST_SERVER=1"})
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer client.Close()

	tools, err := client.ListTools(ctx)
	if err != nil {
		t.Fatalf("ListTools: %v", err)
	}
	if len(tools) != 1 || tools[0].Name != "echo" {
		t.Fatalf("unexpected tools: %+v", tools)
	}

	res, err := client.CallTool(ctx, "echo", map[string]any{"text": "hello"})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if len(res.Content) != 1 || res.Content[0].Text != "hello" {
		t.Fatalf("unexpected result: %+v", res)
	}
}

// runMockServer implements a minimal MCP server: initialize, tools/list, and
// tools/call (echo).
func runMockServer() {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	enc := json.NewEncoder(os.Stdout)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var req struct {
			ID     *int64          `json:"id"`
			Method string          `json:"method"`
		}
		if json.Unmarshal([]byte(line), &req) != nil {
			continue
		}
		// Notifications (no id) get no response.
		if req.ID == nil {
			continue
		}
		var result any
		switch req.Method {
		case "initialize":
			result = map[string]any{
				"protocolVersion": "2024-11-05",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "mock-mcp", "version": "0.1"},
			}
		case "tools/list":
			result = map[string]any{
				"tools": []ToolDef{{
					Name:        "echo",
					Description: "Echoes the input text.",
					InputSchema: map[string]any{"type": "object", "properties": map[string]any{"text": map[string]any{"type": "string"}}, "required": []any{"text"}},
				}},
			}
		case "tools/call":
			result = map[string]any{
				"content": []ContentBlock{{Type: "text", Text: "hello"}},
				"isError": false,
			}
		default:
			enc.Encode(map[string]any{"jsonrpc": "2.0", "id": *req.ID, "error": map[string]any{"code": -32601, "message": fmt.Sprintf("method not found: %s", req.Method)}})
			continue
		}
		enc.Encode(map[string]any{"jsonrpc": "2.0", "id": *req.ID, "result": result})
	}
}

// TestMCPClientCallFailsOnServerExit verifies that when the server process exits
// (closes stdout) while a call is pending, the call returns an error instead of
// hanging forever. The reader goroutine must fail pending channels on EOF.
func TestMCPClientCallFailsOnServerExit(t *testing.T) {
	if os.Getenv("MCP_TEST_SERVER_EXIT") == "1" {
		runMockServerExit()
		return
	}

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	client, err := Start(ctx, exe, nil, []string{"MCP_TEST_SERVER_EXIT=1"})
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer client.Close()

	// Server exits right after initialize. The call below must fail (reader EOF
	// fails pending / marks client closed). Short timeout bounds the test so a
	// regression (hanging call) fails fast instead of stalling the suite.
	callCtx, callCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer callCancel()
	if _, err := client.CallTool(callCtx, "echo", map[string]any{"text": "x"}); err == nil {
		t.Fatal("expected CallTool to fail after server exit, got nil")
	}
	// Client should now be closed; subsequent calls reject fast.
	if _, err := client.CallTool(callCtx, "echo", nil); err == nil {
		t.Fatal("expected second CallTool to fail (client should be closed)")
	}
}

// runMockServerExit handles initialize then exits immediately, closing stdout
// to simulate a server that crashes after startup.
func runMockServerExit() {
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
			continue
		}
		if req.Method == "initialize" {
			enc.Encode(map[string]any{
				"jsonrpc": "2.0", "id": *req.ID,
				"result": map[string]any{
					"protocolVersion": "2024-11-05",
					"capabilities":    map[string]any{},
					"serverInfo":      map[string]any{"name": "mock-mcp", "version": "0.1"},
				},
			})
			return // exit → closes stdout
		}
	}
}
