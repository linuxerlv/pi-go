package main

import (
	"context"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/harness"
)

// harnessRunnerAdapter adapts *harness.AgentHarness to the tui.HarnessRunner
// interface, unwrapping HarnessEvent -> agent.AgentEvent for the TUI.
type harnessRunnerAdapter struct {
	h *harness.AgentHarness
}

func (a harnessRunnerAdapter) Subscribe(handler func(agent.AgentEvent)) func() {
	return a.h.Subscribe(func(e harness.HarnessEvent) error {
		if e.Agent != nil {
			handler(e.Agent)
		}
		return nil
	})
}

func (a harnessRunnerAdapter) Prompt(ctx context.Context, text string) ([]agent.AgentMessage, error) {
	return a.h.Prompt(ctx, text)
}

func (a harnessRunnerAdapter) Phase() string {
	return string(a.h.Phase())
}
