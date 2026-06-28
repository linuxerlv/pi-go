package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
	"github.com/linuxerlv/pi-go/internal/harness"
	"github.com/linuxerlv/pi-go/internal/permission"
	"github.com/linuxerlv/pi-go/internal/tui"
)

// deps bundles all shared state assembled by main() and consumed by the mode
// runners. It is a parameter object: passing one deps value beats threading 10+
// individual arguments through every runner.
type deps struct {
	ctx          context.Context
	model        ai.Model
	prov         ai.Provider
	tools        []agent.AgentTool
	skills       []harness.Skill
	templates    []harness.PromptTemplate
	perm         *permission.Checker
	permMode     string
	system       string
	sessionID    string
	verbose      bool
	prompt       string
	orchestrate  bool
	strategy     string
	maxAgents    int
	useTUI       bool
}

// modeRunner is the strategy interface for a CLI run mode. resolveMode picks
// one based on flags; Run executes it.
type modeRunner interface {
	Run(d deps) error
}

// resolveMode is the factory: it inspects the flags captured in deps and
// returns the appropriate runner. Precedence mirrors the original main()
// if/else chain: TUI > REPL(no prompt) > orchestrate > harness-single > bare.
func resolveMode(d deps) modeRunner {
	switch {
	case d.useTUI:
		return tuiRunner{}
	case d.prompt == "":
		return replRunner{}
	case d.orchestrate:
		return orchestrateRunner{}
	case d.sessionID != "":
		return harnessSingleRunner{}
	default:
		return bareLoopRunner{}
	}
}

// resolveSessionID returns the explicit session id or "default".
func resolveSessionID(d deps) string {
	if d.sessionID != "" {
		return d.sessionID
	}
	return "default"
}

// --- runners ---

type tuiRunner struct{}

func (tuiRunner) Run(d deps) error {
	sid := resolveSessionID(d)
	h := newHarness(sessessionsDir(), sid, d.system, d.model, d.prov, d.tools, d.skills, d.templates, nil)
	runner := harnessRunnerAdapter{h: h}
	mgr := harness.NewSessionManager(sessessionsDir())
	slash := makeTUISlashHandler(h, mgr, sid, d.perm, d.model)
	// Wire the TUI's permission asker into the checker (if --permission set).
	if d.perm != nil {
		prog := tui.NewProgram(runner, d.model.ID, slash)
		perm := permission.New(permission.Options{
			Mode:    permission.Mode(d.permMode),
			Enabled: true,
			Store:   permission.NewStore(),
			Asker:   prog.Asker(),
		})
		h.SetPermission(perm)
		return prog.Run()
	}
	return tui.Run(runner, d.model.ID)
}

// makeTUISlashHandler builds the slash handler the TUI uses for '/'-prefixed
// input. It routes to the shared command registry and returns any output text
// for the TUI to render (it does not itself print; the TUI appends it).
func makeTUISlashHandler(h *harness.AgentHarness, mgr *harness.SessionManager, sid string, perm *permission.Checker, model ai.Model) func(line string) (bool, string) {
	quit := false
	return func(line string) (bool, string) {
		var sb strings.Builder
		sc := SlashContext{
			Harness:    h,
			SessionMgr: mgr,
			SessionID:  sid,
			Permission:  perm,
			Model:       model,
			Out:         &sb,
			Quit:        &quit,
		}
		handled, _ := dispatchSlash(line, sc)
		return handled, strings.TrimRight(sb.String(), "\n")
	}
}

type replRunner struct{}

func (replRunner) Run(d deps) error {
	sid := resolveSessionID(d)
	h := newHarness(sessessionsDir(), sid, d.system, d.model, d.prov, d.tools, d.skills, d.templates, d.perm)
	mgr := harness.NewSessionManager(sessessionsDir())
	runREPL(d.ctx, h, d.verbose, sid, mgr, d.perm, d.model)
	return nil
}

type orchestrateRunner struct{}

func (orchestrateRunner) Run(d deps) error {
	runOrchestrator(d.ctx, d.prompt, d.system, d.strategy, d.maxAgents, d.model, d.prov, d.tools, d.verbose)
	return nil
}

type harnessSingleRunner struct{}

func (harnessSingleRunner) Run(d deps) error {
	emit := func(ev agent.AgentEvent) error {
		printEvent(ev, d.verbose)
		return nil
	}
	runHarness(d.ctx, d.sessionID, d.prompt, d.system, d.model, d.prov, d.tools, d.skills, d.templates, d.perm, emit)
	return nil
}

type bareLoopRunner struct{}

func (bareLoopRunner) Run(d deps) error {
	emit := func(ev agent.AgentEvent) error {
		printEvent(ev, d.verbose)
		return nil
	}
	context_ := agent.AgentContext{
		SystemPrompt: d.system,
		Tools:        d.tools,
	}
	prompts := []agent.AgentMessage{
		ai.UserMessage{Content: d.prompt, Timestamp: ai.Now()},
	}
	config := agent.NewLoopConfig(d.model).
		WithConvertToLlm(agent.DefaultConvertToLlm).
		Build()
	_, err := agent.RunAgentLoop(d.ctx, prompts, context_, config, d.prov, emit)
	return err
}

// runMode dispatches to the resolved runner and exits on error.
func runMode(d deps) {
	runner := resolveMode(d)
	if err := runner.Run(d); err != nil {
		fmt.Fprintf(os.Stderr, "\n[error: %v]\n", err)
		os.Exit(1)
	}
}
