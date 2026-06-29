// Command pi is the entry point for the pi-go agent CLI. It runs the full
// agent loop (prompt -> LLM -> tool calls -> execution -> feedback -> repeat)
// with read and bash tools, across Anthropic or OpenAI-compatible providers.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
	"github.com/linuxerlv/pi-go/internal/ai/provider"
	"github.com/linuxerlv/pi-go/internal/harness"
	"github.com/linuxerlv/pi-go/internal/mcp"
	"github.com/linuxerlv/pi-go/internal/orchestrator"
	"github.com/linuxerlv/pi-go/internal/permission"
	"github.com/linuxerlv/pi-go/internal/tools"
)

func main() {
	loadDotEnv(".env")
	cfg := loadMergedConfig()

	prompt := flag.String("prompt", "", "Prompt to send to the model")
	modelID := flag.String("model", cfg.Model, "Model id (default depends on provider or config)")
	provName := flag.String("provider", cfg.Provider, "Provider: anthropic | openai (auto-detected from env if empty)")
	baseURL := flag.String("base-url", cfg.BaseURL, "Override provider base URL (for OpenAI-compatible endpoints)")
	defaultSystem := "You are a helpful coding agent. Use the read, bash, edit, write, grep, and glob tools to answer questions about the user's codebase. Be concise."
	if cfg.SystemPrompt != "" {
		defaultSystem = cfg.SystemPrompt
	}
	system := flag.String("system", defaultSystem, "System prompt")
	sessionID := flag.String("session", "", "Session id (enables harness mode with jsonl persistence in .pi-go/sessions/)")
	verbose := flag.Bool("verbose", false, "Print all agent events (including tool args) to stderr")
	mcpServers := flag.String("mcp", "", "MCP server to launch as a child process (quoted command, e.g. 'npx -y @modelcontextprotocol/server-filesystem .'). May be set multiple times via config.")
	skillsDir := flag.String("skills-dir", ".skills", "Directory to load skills from (empty to disable)")
	templatesDir := flag.String("templates-dir", "", "Directory to load prompt templates from (empty to disable)")
	useTUI := flag.Bool("tui", false, "Use differential-rendered TUI view instead of streaming print")
	orchestrate := flag.Bool("orchestrate", false, "Use multi-agent orchestrator (decomposes the prompt into subtasks)")
	strategy := flag.String("strategy", "sequential", "Orchestrator strategy: sequential | parallel")
	maxAgents := flag.Int("max-agents", 4, "Maximum concurrent subtask agents (parallel strategy)")
	permMode := flag.String("permission", "", "Permission mode: default | acceptEdits | bypass | plan (empty = permission disabled)")
	flag.Parse()

	if *prompt == "" && flag.NArg() > 0 {
		*prompt = strings.Join(flag.Args(), " ")
	}

	prov, defaultModel, err := resolveProvider(*provName, *baseURL)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(2)
	}
	if *modelID == "" {
		*modelID = defaultModel
	}
	model, ok := prov.GetModel(*modelID)
	if !ok {
		// Register the requested model dynamically so custom endpoints can use
		// arbitrary model ids (e.g. OpenRouter "anthropic/claude-3.5-sonnet").
		model = registerDynamicModel(prov, *modelID)
	}

	cwd, _ := os.Getwd()
	var mcpTools []agent.AgentTool

	// Load optional skills and prompt templates.
	var skills []harness.Skill
	if *skillsDir != "" {
		skills, _ = harness.LoadSkills(*skillsDir)
	}
	var templates []harness.PromptTemplate
	if *templatesDir != "" {
		templates, _ = harness.LoadPromptTemplates(*templatesDir)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Launch optional MCP server(s) and register their tools.
	var mcpClients []*mcp.Client
	if *mcpServers != "" {
		mt, client, err := loadMCPTools(ctx, *mcpServers)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[mcp warning: %v]\n", err)
		}
		mcpTools = append(mcpTools, mt...)
		if client != nil {
			mcpClients = append(mcpClients, client)
		}
	}
	// Ensure MCP child processes are killed on exit; ctx cancellation alone does
	// not terminate them (the MCP client kills the server in Close).
	defer func() {
		for _, c := range mcpClients {
			c.Close()
		}
	}()

	// Build the permission checker when --permission is set.
	var perm *permission.Checker
	if *permMode != "" {
		perm = permission.New(permission.Options{
			Mode:    permission.Mode(*permMode),
			Enabled: true,
			Store:   permission.NewStore(),
			Asker:   makeStdinAsker(),
		})
	}

	// Assemble the tool set for the single-agent paths. The orchestrator gets a
	// ToolFactory instead (below) so each parallel subagent can run with its own
	// tool instances; built-in tools are stateless so this is an isolation
	// extension point rather than a correctness fix.
	agentTools := buildAgentTools(cwd, mcpTools)

	// Assemble the shared deps and dispatch to the selected run mode.
	d := deps{
		ctx:         ctx,
		model:       model,
		prov:        prov,
		tools:       agentTools,
		toolsFactory: func() []agent.AgentTool { return buildAgentTools(cwd, mcpTools) },
		skills:      skills,
		templates:   templates,
		perm:        perm,
		permMode:    *permMode,
		system:      *system,
		sessionID:   *sessionID,
		verbose:     *verbose,
		prompt:      *prompt,
		orchestrate: *orchestrate,
		strategy:    *strategy,
		maxAgents:   *maxAgents,
		useTUI:      *useTUI,
	}
	runMode(d)
	fmt.Fprintln(os.Stderr)
}

// sessessionsDir is the directory for jsonl session files.
func sessessionsDir() string { return ".pi-go/sessions" }

// newHarness builds an AgentHarness backed by a jsonl session.
func newHarness(sessionsDir, sessionID, system string, model ai.Model, prov ai.Provider, agentTools []agent.AgentTool, skills []harness.Skill, templates []harness.PromptTemplate, perm *permission.Checker) *harness.AgentHarness {
	storage, err := harness.NewJsonlStorage(sessionsDir, sessionID)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[session error: %v]\n", err)
		os.Exit(1)
	}
	sess := harness.NewSession(storage)
	return harness.New(harness.Options{
		Provider:        prov,
		Model:           model,
		SystemPrompt:    system,
		Tools:           agentTools,
		Session:         sess,
		Skills:          skills,
		PromptTemplates: templates,
		Permission:      perm,
	})
}

// runHarness runs the prompt through an AgentHarness with a jsonl-persisted
// session, so the conversation can be resumed later with the same --session id.
func runHarness(ctx context.Context, sessionID, prompt, system string, model ai.Model, prov ai.Provider, agentTools []agent.AgentTool, skills []harness.Skill, templates []harness.PromptTemplate, perm *permission.Checker, emit func(agent.AgentEvent) error) {
	h := newHarness(sessessionsDir(), sessionID, system, model, prov, agentTools, skills, templates, perm)
	h.Subscribe(func(e harness.HarnessEvent) error {
		if e.Agent != nil {
			emit(e.Agent)
		}
		return nil
	})
	if _, err := h.Prompt(ctx, prompt); err != nil {
		fmt.Fprintf(os.Stderr, "\n[harness error: %v]\n", err)
		os.Exit(1)
	}
	fmt.Fprintln(os.Stderr)
}

// buildAgentTools constructs the built-in tool set (read/bash/write/edit/grep/
// glob) plus any MCP tools, all bound to cwd. The built-in tools are stateless,
// so constructing fresh instances per call (as the orchestrator's ToolFactory
// does) is safe and isolates future stateful tools per subagent. MCP tools
// share the underlying *mcp.Client (whose call() is concurrency-safe).
func buildAgentTools(cwd string, mcpTools []agent.AgentTool) []agent.AgentTool {
	toolsSlice := []agent.AgentTool{
		tools.NewReadTool(cwd),
		tools.NewBashTool(cwd),
		tools.NewWriteTool(cwd),
		tools.NewEditTool(cwd),
		tools.NewGrepTool(cwd),
		tools.NewGlobTool(cwd),
	}
	toolsSlice = append(toolsSlice, mcpTools...)
	return toolsSlice
}

// runOrchestrator runs the prompt through the multi-agent orchestrator.
func runOrchestrator(ctx context.Context, prompt, system, strategyName string, maxAgents int, model ai.Model, prov ai.Provider, agentTools []agent.AgentTool, perm *permission.Checker, toolFactory func() []agent.AgentTool, verbose bool) {
	var strat orchestrator.Strategy
	switch strategyName {
	case "parallel":
		strat = &orchestrator.ParallelStrategy{MaxConcurrency: maxAgents}
	default:
		strat = &orchestrator.SequentialStrategy{}
	}

	orch := orchestrator.New(orchestrator.Options{
		Provider:     prov,
		Model:        model,
		SystemPrompt: system,
		Tools:        agentTools,
		Strategy:     strat,
		Permission:   perm,
		ToolFactory:  toolFactory,
	})

	fmt.Fprintf(os.Stderr, "[orchestrator] planning subtasks...\n")
	result, err := orch.Run(ctx, prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[orchestrator error: %v]\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "\n=== Orchestrator Result ===\n")
	fmt.Fprintf(os.Stderr, "Task: %s\n", result.Task)
	fmt.Fprintf(os.Stderr, "Subtasks: %d\n\n", len(result.SubTasks))
	for i, st := range result.SubTasks {
		fmt.Fprintf(os.Stderr, "--- Subtask %d: %s ---\n", i+1, st.ID)
		if verbose {
			fmt.Fprintf(os.Stderr, "%s\n", st.Description)
			if i < len(result.Results) {
				fmt.Fprintf(os.Stderr, "Result: %s\n", result.Results[i].Output)
			}
		}
	}
	fmt.Fprintf(os.Stderr, "\nFinal Answer:\n%s\n", result.FinalAnswer)
}

// loadMCPTools launches an MCP server (quoted string) and returns its tools
// plus the client (caller must Close it to reap the child process).
func loadMCPTools(ctx context.Context, command string) ([]agent.AgentTool, *mcp.Client, error) {
	parts, err := splitShellArgs(command)
	if err != nil {
		return nil, nil, err
	}
	if len(parts) == 0 {
		return nil, nil, fmt.Errorf("empty mcp command")
	}
	client, err := mcp.Start(ctx, parts[0], parts[1:], nil)
	if err != nil {
		return nil, nil, err
	}
	defs, err := client.ListTools(ctx)
	if err != nil {
		client.Close()
		return nil, nil, err
	}
	var out []agent.AgentTool
	for _, d := range defs {
		out = append(out, mcp.NewMCPTool(client, d))
	}
	return out, client, nil
}

// splitShellArgs splits a command string on whitespace, honoring double quotes.
func splitShellArgs(s string) ([]string, error) {
	var out []string
	var cur strings.Builder
	inQuote := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c == '"':
			inQuote = !inQuote
		case (c == ' ' || c == '\t' || c == '\n') && !inQuote:
			if cur.Len() > 0 {
				out = append(out, cur.String())
				cur.Reset()
			}
		default:
			cur.WriteByte(c)
		}
	}
	if inQuote {
		return nil, fmt.Errorf("unterminated quote in command")
	}
	if cur.Len() > 0 {
		out = append(out, cur.String())
	}
	return out, nil
}

// resolveProvider picks the provider by name, or auto-detects from environment
// variables. Returns the provider and a sensible default model id.
func resolveProvider(name, baseURL string) (ai.Provider, string, error) {
	name = strings.ToLower(name)
	if name == "" {
		// Auto-detect: prefer Anthropic if its env is set, else OpenAI.
		if os.Getenv("ANTHROPIC_API_KEY") != "" || os.Getenv("ANTHROPIC_AUTH_TOKEN") != "" {
			name = "anthropic"
		} else if os.Getenv("OPENAI_API_KEY") != "" || os.Getenv("OPENAI_AUTH_TOKEN") != "" {
			name = "openai"
		} else {
			return nil, "", fmt.Errorf("no provider configured: set ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN or OPENAI_API_KEY/OPENAI_AUTH_TOKEN, or use --provider")
		}
	}

	switch name {
	case "anthropic":
		prov, err := provider.NewAnthropicFromEnv()
		if err != nil {
			return nil, "", err
		}
		return prov, "claude-haiku-4-5", nil
	case "openai":
		apiKey := openAIKeyFromEnv()
		if apiKey == "" {
			return nil, "", fmt.Errorf("OPENAI_API_KEY (or OPENAI_AUTH_TOKEN) is not set")
		}
		url := baseURL
		if url == "" {
			url = os.Getenv("OPENAI_BASE_URL")
		}
		prov := provider.NewOpenAI(apiKey, url)
		return prov, "gpt-4o-mini", nil
	case "openai-responses":
		apiKey := openAIKeyFromEnv()
		if apiKey == "" {
			return nil, "", fmt.Errorf("OPENAI_API_KEY (or OPENAI_AUTH_TOKEN) is not set")
		}
		url := baseURL
		if url == "" {
			url = os.Getenv("OPENAI_BASE_URL")
		}
		prov := provider.NewOpenAIResponses(apiKey, url)
		return prov, "gpt-4o-mini", nil
	}
	return nil, "", fmt.Errorf("unknown provider: %s (use anthropic, openai, or openai-responses)", name)
}

// registerDynamicModel adds an unknown model id to the provider so the agent
// loop can run against custom endpoints with arbitrary model names.
func registerDynamicModel(prov ai.Provider, id string) ai.Model {
	switch p := prov.(type) {
	case *provider.Anthropic:
		m := ai.Model{ID: id, Name: id, API: ai.APIAnthropicMessages, Provider: "anthropic", ContextWindow: 200_000, MaxTokens: 32_000, Input: []string{"text", "image"}}
		p.RegisterModel(m)
		return m
	case *provider.OpenAI:
		m := ai.Model{ID: id, Name: id, API: ai.APIOpenAICompletions, Provider: "openai", ContextWindow: 128_000, MaxTokens: 16_384, Input: []string{"text", "image"}}
		p.RegisterModel(m)
		return m
	case *provider.OpenAIResponses:
		m := ai.Model{ID: id, Name: id, API: ai.APIOpenAIResponses, Provider: "openai-responses", ContextWindow: 128_000, MaxTokens: 16_384, Input: []string{"text", "image"}}
		p.RegisterModel(m)
		return m
	}
	return ai.Model{ID: id, Name: id}
}

// openAIKeyFromEnv returns the OpenAI API key from OPENAI_API_KEY, falling back
// to OPENAI_AUTH_TOKEN.
func openAIKeyFromEnv() string {
	k := os.Getenv("OPENAI_API_KEY")
	if k == "" {
		k = os.Getenv("OPENAI_AUTH_TOKEN")
	}
	return k
}
