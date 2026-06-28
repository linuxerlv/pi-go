package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/linuxerlv/pi-go/internal/ai"
	"github.com/linuxerlv/pi-go/internal/harness"
	"github.com/linuxerlv/pi-go/internal/permission"
)

// runREPL starts an interactive read-eval-print loop. Each input line is sent
// as a prompt to the harness; the conversation persists across turns via the
// session. Slash commands are handled by the shared command registry.
func runREPL(ctx context.Context, h *harness.AgentHarness, verbose bool, sessionID string, mgr *harness.SessionManager, perm *permission.Checker, model ai.Model) {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	// Subscribe once; the handler renders every agent event for the session.
	h.Subscribe(func(e harness.HarnessEvent) error {
		if e.Agent != nil {
			printEvent(e.Agent, verbose)
		}
		return nil
	})

	quit := false
	sc := SlashContext{
		Harness:    h,
		SessionMgr: mgr,
		SessionID:  sessionID,
		Permission: perm,
		Model:      model,
		Out:        os.Stderr,
		Quit:       &quit,
	}

	fmt.Println("pi-go interactive session. Type /help for commands, /exit to quit.")
	for {
		if quit {
			return
		}
		fmt.Fprint(os.Stderr, "\n> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if line[0] == '/' {
			dispatchSlash(line, sc)
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
