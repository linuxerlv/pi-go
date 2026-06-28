// Package permission implements a tool-execution permission model for pi-go,
// inspired by Claude Code's permission modes. A Checker decides, for each tool
// call, whether to allow it, deny it, or ask the user. Decisions can be
// persisted (allow-always) and a permission mode widens or narrows the default
// policy.
package permission

import (
	"context"
	"path/filepath"
	"regexp"
	"strings"
)

// Mode controls how aggressively the checker asks the user.
type Mode string

const (
	// ModeDefault asks before destructive or write operations.
	ModeDefault Mode = "default"
	// ModeAcceptEdits auto-allows file edits/writes but still asks for bash.
	ModeAcceptEdits Mode = "acceptEdits"
	// ModeBypass allows everything without asking.
	ModeBypass Mode = "bypass"
	// ModePlan is read-only: any write/destructive op is denied.
	ModePlan Mode = "plan"
)

// Decision is the outcome of a permission check.
type Decision string

const (
	// DecisionAllow lets the tool run (this time only).
	DecisionAllow Decision = "allow"
	// DecisionDeny blocks the tool; it receives an error result.
	DecisionDeny Decision = "deny"
	// DecisionAsk means the checker wants the user to confirm.
	DecisionAsk Decision = "ask"
)

// Asker is a callback that prompts the user for a decision. It returns one of
// "allow", "allow-always" (persist), or "deny". The implementer renders the
// prompt (e.g. a bubbletea modal or an stdin readline) and blocks until the
// user answers.
type Asker func(ctx context.Context, prompt string) (string, error)

// Rule is a single allow/deny rule matching a tool + args.
type Rule struct {
	// Tool is the tool name ("bash", "write", ...) or "*" for any.
	Tool string
	// Kind is "allow" or "deny".
	Kind string
	// ArgPattern is a regex matched against a stringified arg representation:
	// for bash it is the command string; for file tools it is the path.
	ArgPattern *regexp.Regexp
	// PathPrefix matches file-tool paths by prefix (resolved absolute).
	PathPrefix string
}

// Checker holds the active permission mode, rules, and persistence.
type Checker struct {
	mode    Mode
	rules   []Rule
	asker   Asker
	store   *Store
	enabled bool
}

// Options configures a Checker.
type Options struct {
	Mode    Mode
	Rules   []Rule
	Asker   Asker
	Store   *Store // nil disables persistence
	Enabled bool
}

// New constructs a Checker with the built-in policy rules applied.
func New(opts Options) *Checker {
	if opts.Mode == "" {
		opts.Mode = ModeDefault
	}
	rules := append([]Rule(nil), builtinPolicy()...)
	rules = append(rules, opts.Rules...)
	return &Checker{
		mode:    opts.Mode,
		rules:   rules,
		asker:   opts.Asker,
		store:   opts.Store,
		enabled: opts.Enabled,
	}
}

// Mode returns the current mode.
func (c *Checker) Mode() Mode { return c.mode }

// SetMode changes the permission mode at runtime.
func (c *Checker) SetMode(m Mode) { c.mode = m }

// Enabled reports whether permission checks are active.
func (c *Checker) Enabled() bool { return c.enabled }

// CheckArgs is the argument bundle passed to Check. Extra fields can be added
// later without breaking callers.
type CheckArgs struct {
	ToolName string
	// Command is the bash command string (for bash).
	Command string
	// Path is the file path (for read/write/edit/grep/glob).
	Path string
}

// Check decides what to do with a tool call. It applies mode + rules + store,
// then asks the user if still undecided (and an asker is configured).
// Returns (Decision, reason). DecisionAllow means proceed.
func (c *Checker) Check(ctx context.Context, args CheckArgs) (Decision, string) {
	if !c.enabled || c.mode == ModeBypass {
		return DecisionAllow, ""
	}

	// 1. Persistent allow-always rules win.
	if c.store != nil {
		if c.store.IsAllowedAlways(args.ToolName, args) {
			return DecisionAllow, ""
		}
	}

	// 2. Built-in + custom rules.
	decision, reason := c.applyRules(args)
	if decision == DecisionDeny {
		return DecisionDeny, reason
	}
	if decision == DecisionAllow {
		return DecisionAllow, ""
	}

	// 3. Mode-based default.
	if c.mode == ModePlan {
		if isWriteTool(args.ToolName) {
			return DecisionDeny, "plan mode: write operations are blocked"
		}
		return DecisionAllow, ""
	}
	if c.mode == ModeAcceptEdits && isFileEditTool(args.ToolName) {
		return DecisionAllow, ""
	}

	// 4. Ask the user (only for operations that need confirmation).
	if !needsAsk(args) {
		return DecisionAllow, ""
	}
	if c.asker == nil {
		// No asker configured: deny destructive, allow the rest.
		if isDestructive(args) {
			return DecisionDeny, "destructive operation requires confirmation (no asker)"
		}
		return DecisionAllow, ""
	}

	prompt := formatPrompt(args)
	answer, err := c.asker(ctx, prompt)
	if err != nil || answer == "deny" {
		return DecisionDeny, "denied by user"
	}
	if answer == "allow-always" && c.store != nil {
		_ = c.store.AddAllowAlways(args.ToolName, args)
	}
	return DecisionAllow, ""
}

// applyRules evaluates allow/deny rules in order; first match wins.
func (c *Checker) applyRules(args CheckArgs) (Decision, string) {
	for _, r := range c.rules {
		if r.Tool != "*" && r.Tool != args.ToolName {
			continue
		}
		matched := false
		if r.ArgPattern != nil {
			target := args.Command
			if target == "" {
				target = args.Path
			}
			if r.ArgPattern.MatchString(target) {
				matched = true
			}
		}
		if r.PathPrefix != "" && args.Path != "" {
			abs, err := filepath.Abs(args.Path)
			if err == nil && strings.HasPrefix(abs, r.PathPrefix) {
				matched = true
			}
		}
		if !matched {
			continue
		}
		if r.Kind == "deny" {
			return DecisionDeny, "blocked by deny rule"
		}
		return DecisionAllow, ""
	}
	return DecisionAsk, ""
}

// needsAsk reports whether an operation requires user confirmation in default
// mode (after rules and mode didn't already decide).
func needsAsk(args CheckArgs) bool {
	if args.ToolName == "bash" {
		return true // bash always asks in default mode unless allow-always
	}
	if isWriteTool(args.ToolName) {
		return true
	}
	return false
}

func isWriteTool(name string) bool {
	return name == "write" || name == "edit" || name == "bash"
}

func isFileEditTool(name string) bool {
	return name == "write" || name == "edit"
}

func isDestructive(args CheckArgs) bool {
	if args.ToolName == "bash" {
		return destructiveBashRe.MatchString(args.Command)
	}
	return false
}

func formatPrompt(args CheckArgs) string {
	if args.ToolName == "bash" {
		return "run bash: " + args.Command
	}
	if args.Path != "" {
		return args.ToolName + " " + args.Path
	}
	return args.ToolName
}
