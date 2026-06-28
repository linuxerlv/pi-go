package tui

import (
	"context"
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/linuxerlv/pi-go/internal/agent"
)

// Program wraps a tea.Program plus a permission asker wired to it. Build it
// with NewProgram, wire the asker into the permission checker, then call Run.
type Program struct {
	prog  *tea.Program
	model *Model
}

// NewProgram constructs the TUI program (alt screen) without starting it, so
// the caller can wire the permission asker into a harness before running.
func NewProgram(h HarnessRunner, modelName string) *Program {
	m := NewModel(h, modelName)
	p := tea.NewProgram(m, tea.WithAltScreen())
	// Subscribe to agent events; forward each into the tea loop.
	h.Subscribe(func(ev agent.AgentEvent) {
		p.Send(agentEventMsg{event: ev})
	})
	return &Program{prog: p, model: &m}
}

// Asker returns a permission.Asker that routes prompts through this TUI as a
// modal and blocks until the user answers.
func (pr *Program) Asker() func(context.Context, string) (string, error) {
	return func(ctx context.Context, prompt string) (string, error) {
		replyCh := make(chan string, 1)
		pr.prog.Send(permRequestMsg{prompt: prompt, replyCh: replyCh})
		select {
		case ans := <-replyCh:
			return ans, nil
		case <-ctx.Done():
			return "deny", ctx.Err()
		}
	}
}

// Run starts the event loop and blocks until the user quits.
func (pr *Program) Run() error {
	_, err := pr.prog.Run()
	return err
}

// Run is a convenience that builds and runs a program in one call (no permission
// asker wiring). Used when permission is disabled.
func Run(h HarnessRunner, modelName string) error {
	pr := NewProgram(h, modelName)
	return pr.Run()
}

// reportErr is a small helper for CLI error reporting.
func reportErr(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "[tui error:", err, "]")
	}
}
