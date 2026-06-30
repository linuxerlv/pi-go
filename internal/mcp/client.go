// Package mcp is a minimal Model Context Protocol (MCP) client over stdio.
// It spawns an MCP server as a child process, performs the initialize
// handshake, lists tools, and forwards tool calls. Each MCP tool is exposed as
// an agent.AgentTool so the agent loop can invoke it like any built-in tool.
//
// This is a pi-go addition (pi itself does not ship MCP); it lets pi-go reuse
// the ecosystem of MCP tool servers.
package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"sync"
	"sync/atomic"
)

// Client is a stdio JSON-RPC client to an MCP server child process.
type Client struct {
	cmd    *exec.Cmd
	stdin  chan []byte
	stdout chan []byte
	nextID atomic.Int64
	mu     sync.Mutex
	pending map[int64]chan json.RawMessage
	closed bool
}

// ToolDef is an MCP tool definition.
type ToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

// CallResult is the result of an MCP tools/call.
type CallResult struct {
	Content []ContentBlock `json:"content"`
	IsError bool           `json:"isError"`
}

// ContentBlock is one block of an MCP tool result.
type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// Start launches an MCP server command and performs the initialize handshake.
// The command is run with the given args and env. Returns a ready client.
func Start(ctx context.Context, command string, args []string, env []string) (*Client, error) {
	cmd := exec.Command(command, args...)
	if len(env) > 0 {
		cmd.Env = env
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start mcp server: %w", err)
	}
	c := &Client{
		cmd:     cmd,
		stdin:   make(chan []byte, 64),
		pending: map[int64]chan json.RawMessage{},
	}
	// Writer goroutine: frames messages as Content-Length headers (MCP stdio
	// transport uses newline-delimited JSON by default; this implementation
	// uses newline-delimited JSON which most stdio servers accept).
	go func() {
		for msg := range c.stdin {
			if _, err := stdin.Write(append(msg, '\n')); err != nil {
				return
			}
		}
		stdin.Close()
	}()
	// Reader goroutine: parses responses and routes by id.
	go func() {
		scanner := bufio.NewScanner(stdout)
		scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}
			var envelope struct {
				ID     *int64           `json:"id"`
				Result json.RawMessage  `json:"result"`
				Error  *RPCError        `json:"error"`
			}
			if json.Unmarshal(line, &envelope) != nil {
				continue
			}
			if envelope.ID == nil {
				continue // notification
			}
			c.mu.Lock()
			ch, ok := c.pending[*envelope.ID]
			delete(c.pending, *envelope.ID)
			c.mu.Unlock()
			if !ok {
				continue
			}
			if envelope.Error != nil {
				ch <- nil
				close(ch)
				continue
			}
			ch <- envelope.Result
			close(ch)
		}
		// stdout closed (server exited/crashed). Fail every pending call so a
		// caller blocked in `call`'s <-ch receives an error instead of hanging
		// forever. Mark closed so new calls reject fast.
		c.mu.Lock()
		c.closed = true
		for id, ch := range c.pending {
			ch <- nil
			close(ch)
			delete(c.pending, id)
		}
		c.mu.Unlock()
	}()

	// initialize handshake.
	if _, err := c.call(ctx, "initialize", map[string]any{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]any{"name": "pi-go", "version": "0.1"},
	}); err != nil {
		c.Close()
		return nil, fmt.Errorf("mcp initialize: %w", err)
	}
	// initialized notification (no id).
	c.notify("notifications/initialized", map[string]any{})
	return c, nil
}

// RPCError is a JSON-RPC error.
type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// call sends a JSON-RPC request and awaits its result.
func (c *Client) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := c.nextID.Add(1)
	req := map[string]any{"jsonrpc": "2.0", "id": id, "method": method}
	if params != nil {
		req["params"] = params
	}
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	ch := make(chan json.RawMessage, 1)
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil, fmt.Errorf("mcp client closed")
	}
	c.pending[id] = ch
	c.mu.Unlock()

	select {
	case c.stdin <- b:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	select {
	case res := <-ch:
		if res == nil {
			return nil, fmt.Errorf("mcp call %s returned an error", method)
		}
		return res, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (c *Client) notify(method string, params any) {
	req := map[string]any{"jsonrpc": "2.0", "method": method}
	if params != nil {
		req["params"] = params
	}
	b, _ := json.Marshal(req)
	select {
	case c.stdin <- b:
	default:
	}
}

// ListTools returns the tools exposed by the server.
func (c *Client) ListTools(ctx context.Context) ([]ToolDef, error) {
	res, err := c.call(ctx, "tools/list", nil)
	if err != nil {
		return nil, err
	}
	var wrapper struct {
		Tools []ToolDef `json:"tools"`
	}
	if err := json.Unmarshal(res, &wrapper); err != nil {
		return nil, fmt.Errorf("parse tools/list: %w", err)
	}
	return wrapper.Tools, nil
}

// CallTool invokes a tool by name with the given arguments.
func (c *Client) CallTool(ctx context.Context, name string, arguments map[string]any) (CallResult, error) {
	res, err := c.call(ctx, "tools/call", map[string]any{"name": name, "arguments": arguments})
	if err != nil {
		return CallResult{}, err
	}
	var result CallResult
	if err := json.Unmarshal(res, &result); err != nil {
		return CallResult{}, fmt.Errorf("parse tools/call: %w", err)
	}
	return result, nil
}

// Close stops the server process.
func (c *Client) Close() error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	c.mu.Unlock()
	close(c.stdin)
	if c.cmd != nil && c.cmd.Process != nil {
		_ = c.cmd.Process.Kill()
		_ = c.cmd.Wait()
	}
	return nil
}
