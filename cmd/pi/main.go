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
	"github.com/linuxerlv/pi-go/internal/tools"
)

func main() {
	loadDotEnv(".env")

	prompt := flag.String("prompt", "", "Prompt to send to the model")
	modelID := flag.String("model", "", "Model id (default depends on provider)")
	provName := flag.String("provider", "", "Provider: anthropic | openai (auto-detected from env if empty)")
	baseURL := flag.String("base-url", "", "Override provider base URL (for OpenAI-compatible endpoints)")
	system := flag.String("system", "You are a helpful coding agent. Use the read and bash tools to answer questions about the user's codebase. Be concise.", "System prompt")
	sessionID := flag.String("session", "", "Session id (enables harness mode with jsonl persistence in .pi-go/sessions/)")
	verbose := flag.Bool("verbose", false, "Print all agent events (including tool args) to stderr")
	flag.Parse()

	if *prompt == "" && flag.NArg() > 0 {
		*prompt = strings.Join(flag.Args(), " ")
	}
	if *prompt == "" {
		fmt.Fprintln(os.Stderr, `usage: pi --prompt "..." [--provider anthropic|openai] [--model ...] [--base-url ...]`)
		os.Exit(2)
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
	agentTools := []agent.AgentTool{
		tools.NewReadTool(cwd),
		tools.NewBashTool(cwd),
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	emit := func(ev agent.AgentEvent) error {
		printEvent(ev, *verbose)
		return nil
	}

	if *sessionID != "" {
		// Harness mode: stateful, jsonl-persisted session.
		runHarness(ctx, *sessionID, *prompt, *system, model, prov, agentTools, emit)
		return
	}

	// Bare loop mode (M2 behavior).
	context_ := agent.AgentContext{
		SystemPrompt: *system,
		Tools:        agentTools,
	}
	prompts := []agent.AgentMessage{
		ai.UserMessage{Content: *prompt, Timestamp: ai.Now()},
	}
	config := agent.AgentLoopConfig{
		Model:        model,
		ConvertToLlm: agent.DefaultConvertToLlm,
	}

	if _, err := agent.RunAgentLoop(ctx, prompts, context_, config, prov, emit); err != nil {
		fmt.Fprintf(os.Stderr, "\n[loop error: %v]\n", err)
		os.Exit(1)
	}
	fmt.Fprintln(os.Stderr)
}

// runHarness runs the prompt through an AgentHarness with a jsonl-persisted
// session, so the conversation can be resumed later with the same --session id.
func runHarness(ctx context.Context, sessionID, prompt, system string, model ai.Model, prov ai.Provider, agentTools []agent.AgentTool, emit func(agent.AgentEvent) error) {
	sessionsDir := ".pi-go/sessions"
	storage, err := harness.NewJsonlStorage(sessionsDir, sessionID)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\n[session error: %v]\n", err)
		os.Exit(1)
	}
	sess := harness.NewSession(storage)
	h := harness.New(harness.Options{
		Provider:     prov,
		Model:        model,
		SystemPrompt: system,
		Tools:        agentTools,
		Session:      sess,
	})
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
		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_AUTH_TOKEN")
		}
		if apiKey == "" {
			return nil, "", fmt.Errorf("OPENAI_API_KEY (or OPENAI_AUTH_TOKEN) is not set")
		}
		url := baseURL
		if url == "" {
			url = os.Getenv("OPENAI_BASE_URL")
		}
		prov := provider.NewOpenAI(apiKey, url)
		return prov, "gpt-4o-mini", nil
	}
	return nil, "", fmt.Errorf("unknown provider: %s (use anthropic or openai)", name)
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
	}
	return ai.Model{ID: id, Name: id}
}
