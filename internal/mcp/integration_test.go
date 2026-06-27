//go:build mcp_integration

// The mcp_integration build tag gates this test so it only runs when node is
// available and the user explicitly opts in via: go test -tags mcp_integration
package mcp

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestMCPClientRealEchoServer(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not in PATH")
	}
	server := filepath.Join("testdata", "echo_server.js")
	if _, err := os.Stat(server); err != nil {
		t.Skip("echo_server.js not found: " + server)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	client, err := Start(ctx, node, []string{server}, nil)
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
	if tools[0].Description != "Echoes text back." {
		t.Fatalf("unexpected description: %s", tools[0].Description)
	}

	res, err := client.CallTool(ctx, "echo", map[string]any{"text": "hello-mcp"})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if len(res.Content) != 1 || res.Content[0].Text != "hello-mcp" {
		t.Fatalf("unexpected result: %+v", res)
	}
}
