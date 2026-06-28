package main

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/linuxerlv/pi-go/internal/ai"
	"github.com/linuxerlv/pi-go/internal/harness"
	"github.com/linuxerlv/pi-go/internal/permission"
)

// SlashContext is passed to each slash command's Run.
type SlashContext struct {
	Harness     *harness.AgentHarness
	SessionMgr  *harness.SessionManager
	SessionID   string
	Permission  *permission.Checker
	Model       ai.Model
	Out         io.Writer
	Quit        *bool // set true to exit the REPL/TUI
}

// Command is a registered slash command.
type Command struct {
	Name        string
	Description string
	Usage       string // args description, e.g. "<id>" or "<path.html|.jsonl>"; "" if no args
	Run         func(args []string, sc SlashContext) error
}

// commands returns all registered slash commands.
func commands() []Command {
	return []Command{
		{Name: "/help", Description: "List slash commands", Run: cmdHelp},
		{Name: "/quit", Description: "Exit pi-go", Run: cmdQuit},
		{Name: "/exit", Description: "Exit pi-go", Run: cmdQuit},
		{Name: "/sessions", Description: "List sessions", Run: cmdSessions},
		{Name: "/resume", Description: "Switch to a session", Usage: "<id>", Run: cmdResume},
		{Name: "/new", Description: "Start a new session", Usage: "[id]", Run: cmdNew},
		{Name: "/fork", Description: "Fork the session at the current point", Run: cmdFork},
		{Name: "/name", Description: "Set a session label", Usage: "<text>", Run: cmdName},
		{Name: "/session", Description: "Show current session info", Run: cmdSession},
		{Name: "/export", Description: "Export session", Usage: "<path.html|.jsonl>", Run: cmdExport},
		{Name: "/import", Description: "Import a jsonl session", Usage: "<path> [id]", Run: cmdImport},
		{Name: "/model", Description: "Switch model", Usage: "<id>", Run: cmdModel},
		{Name: "/permission", Description: "Set permission mode", Usage: "default|acceptEdits|bypass|plan", Run: cmdPermission},
		{Name: "/compact", Description: "Manually compact context (if configured)", Run: cmdCompact},
		{Name: "/tools", Description: "List available tools", Run: cmdTools},
	}
}

// dispatchSlash executes a slash command line. Returns (handled, shouldQuit).
func dispatchSlash(line string, sc SlashContext) (bool, bool) {
	parts := strings.Fields(line)
	if len(parts) == 0 {
		return false, false
	}
	name := parts[0]
	for _, c := range commands() {
		if c.Name == name {
			err := c.Run(parts[1:], sc)
			if err != nil {
				fmt.Fprintf(sc.Out, "[error: %v]\n", err)
			}
			return true, *sc.Quit
		}
	}
	fmt.Fprintf(sc.Out, "unknown command: %s (try /help)\n", name)
	return true, false
}

func cmdHelp(args []string, sc SlashContext) error {
	fmt.Fprintln(sc.Out, "commands:")
	for _, c := range commands() {
		if c.Usage != "" {
			fmt.Fprintf(sc.Out, "  %-12s %s  %s\n", c.Name, c.Usage, c.Description)
		} else {
			fmt.Fprintf(sc.Out, "  %-12s %s\n", c.Name, c.Description)
		}
	}
	return nil
}

func cmdQuit(args []string, sc SlashContext) error {
	*sc.Quit = true
	return nil
}

func cmdSessions(args []string, sc SlashContext) error {
	if sc.SessionMgr == nil {
		return fmt.Errorf("session manager not configured")
	}
	infos, err := sc.SessionMgr.List()
	if err != nil {
		return err
	}
	if len(infos) == 0 {
		fmt.Fprintln(sc.Out, "(no sessions)")
		return nil
	}
	for _, info := range infos {
		label := ""
		if info.Label != "" {
			label = "  [" + info.Label + "]"
		}
		marker := ""
		if info.ID == sc.SessionID {
			marker = " *"
		}
		fmt.Fprintf(sc.Out, "  %s%s  (%d msgs, %s)%s\n", info.ID, marker, info.MessageCount, info.CreatedAt, label)
	}
	return nil
}

func cmdResume(args []string, sc SlashContext) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: /resume <session-id>")
	}
	if _, err := sc.SessionMgr.Open(args[0]); err != nil {
		return fmt.Errorf("cannot open session %s: %w", args[0], err)
	}
	fmt.Fprintf(sc.Out, "session %s is available; restart with --session %s to switch into it\n", args[0], args[0])
	return nil
}

func cmdNew(args []string, sc SlashContext) error {
	id := ""
	if len(args) > 0 {
		id = args[0]
	}
	_, err := sc.SessionMgr.Create(id)
	if err != nil {
		return err
	}
	if id == "" {
		fmt.Fprintln(sc.Out, "created a new session; restart without --session to use the default, or pass the generated id")
	} else {
		fmt.Fprintf(sc.Out, "created session %s; restart with --session %s\n", id, id)
	}
	return nil
}

func cmdFork(args []string, sc SlashContext) error {
	if sc.Harness == nil {
		return fmt.Errorf("no active session")
	}
	leaf, err := sc.Harness.Session().Fork("")
	if err != nil {
		return err
	}
	fmt.Fprintf(sc.Out, "forked session at leaf %s\n", leaf)
	return nil
}

func cmdName(args []string, sc SlashContext) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: /name <label text>")
	}
	label := strings.Join(args, " ")
	return sc.Harness.Session().SetLabel(label)
}

func cmdSession(args []string, sc SlashContext) error {
	sess := sc.Harness.Session()
	ctx := sess.BuildContext()
	fmt.Fprintf(sc.Out, "session: %s\n", sess.Metadata().ID)
	fmt.Fprintf(sc.Out, "messages: %d\n", len(ctx.Messages))
	if ctx.Model != nil {
		fmt.Fprintf(sc.Out, "model: %s/%s\n", ctx.Model.Provider, ctx.Model.ModelID)
	}
	if sc.Harness.Permission() != nil {
		fmt.Fprintf(sc.Out, "permission: %s\n", sc.Harness.Permission().Mode())
	}
	return nil
}

func cmdExport(args []string, sc SlashContext) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: /export <path.html|.jsonl>")
	}
	path := args[0]
	sess := sc.Harness.Session()
	if strings.HasSuffix(path, ".jsonl") {
		return harness.ExportJSONL(sess, path)
	}
	return harness.ExportHTML(sess, path)
}

func cmdImport(args []string, sc SlashContext) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: /import <path> [id]")
	}
	id := ""
	if len(args) > 1 {
		id = args[1]
	}
	_, err := harness.ImportJSONL(sc.SessionMgr, args[0], id)
	if err != nil {
		return err
	}
	fmt.Fprintln(sc.Out, "imported session")
	return nil
}

func cmdModel(args []string, sc SlashContext) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: /model <model-id>")
	}
	fmt.Fprintf(sc.Out, "model switch requires restart with --model %s (in-session model change via harness.SetModel is supported but the provider must resolve it)\n", args[0])
	return nil
}

func cmdPermission(args []string, sc SlashContext) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: /permission default|acceptEdits|bypass|plan")
	}
	if sc.Permission == nil {
		return fmt.Errorf("permission not enabled (start with --permission)")
	}
	sc.Permission.SetMode(permission.Mode(args[0]))
	fmt.Fprintf(sc.Out, "permission mode set to %s\n", args[0])
	return nil
}

func cmdCompact(args []string, sc SlashContext) error {
	// Manual compaction entry: the harness auto-compacts when configured; here
	// we just report status. A full manual compact would call the harness
	// compaction API, which is wired via CompactionConfig.
	fmt.Fprintln(sc.Out, "compaction is automatic when configured via CompactionConfig; no manual action taken")
	return nil
}

func cmdTools(args []string, sc SlashContext) error {
	fmt.Fprintln(sc.Out, "tools: read, bash, write, edit, grep, glob (+ any MCP tools)")
	return nil
}

// keep imports used
var _ = context.Background
