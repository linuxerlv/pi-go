package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/linuxerlv/pi-go/internal/harness"
)

// runREPL starts an interactive read-eval-print loop. Each input line is sent
// as a prompt to the harness; the conversation persists across turns via the
// session. Slash commands: /exit /quit /clear /help.
func runREPL(ctx context.Context, h *harness.AgentHarness, verbose bool) {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	// Subscribe once; the handler renders every agent event for the session.
	h.Subscribe(func(e harness.HarnessEvent) error {
		if e.Agent != nil {
			printEvent(e.Agent, verbose)
		}
		return nil
	})

	fmt.Println("pi-go interactive session. Type /help for commands, /exit to quit.")
	for {
		fmt.Fprint(os.Stderr, "\n> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if line[0] == '/' {
			if handleSlashCommand(line, h) {
				return // exit requested
			}
			continue
		}

		if _, err := h.Prompt(ctx, line); err != nil {
			fmt.Fprintf(os.Stderr, "[error: %v]\n", err)
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "[input error: %v]\n", err)
	}
}

// handleSlashCommand returns true if the REPL should exit.
func handleSlashCommand(line string, h *harness.AgentHarness) bool {
	parts := strings.Fields(line)
	cmd := parts[0]
	switch cmd {
	case "/exit", "/quit":
		return true
	case "/clear":
		// Clearing is approximated by starting a new session; the caller can
		// restart with a fresh --session id. For now, report state.
		fmt.Fprintln(os.Stderr, "(session retained; restart with a new --session to clear)")
	case "/help":
		fmt.Fprintln(os.Stderr, "commands:\n  /exit, /quit  leave the REPL\n  /clear        (restart with new --session to clear)\n  /help         this message\notherwise: type a prompt and press Enter")
	case "/tools":
		// Listing tools would need access to the tool list; print phase.
		fmt.Fprintf(os.Stderr, "(phase: %s)\n", h.Phase())
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s (try /help)\n", cmd)
	}
	return false
}
