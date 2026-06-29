package harness

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
	"github.com/linuxerlv/pi-go/internal/permission"
)

// Phase is the harness run state.
type Phase string

const (
	PhaseIdle          Phase = "idle"
	PhaseTurn          Phase = "turn"
	PhaseCompaction    Phase = "compaction"
	PhaseBranchSummary Phase = "branch_summary"
)

// HarnessEvent is emitted by the harness to subscribers. It wraps an agent
// event plus harness-level events.
type HarnessEvent struct {
	Agent agent.AgentEvent // nil for harness-only events
	Phase Phase
}

// EventHandler is a subscriber callback.
type EventHandler func(HarnessEvent) error

// Options configures an AgentHarness.
type Options struct {
	Provider     ai.Provider
	Model        ai.Model
	ThinkingLevel ai.ThinkingLevel
	SystemPrompt string
	Tools        []agent.AgentTool
	Session      *Session
	// Skills available for explicit Skill(name) invocation and model-visible
	// listing in the system prompt.
	Skills []Skill
	// PromptTemplates available for explicit PromptFromTemplate(name, args).
	PromptTemplates []PromptTemplate
	// Compaction configures automatic context-window compaction. Nil disables
	// automatic compaction (the default).
	Compaction *CompactionConfig
	// Permission, if set, is consulted before each tool call via the agent
	// loop's BeforeToolCall hook. Nil disables permission checks.
	Permission *permission.Checker
}

// AgentHarness is a stateful wrapper around the agent loop with session
// persistence, steer/follow-up queues, and event subscription. It mirrors
// @earendil-works/pi-agent-core's AgentHarness.
type AgentHarness struct {
	mu sync.Mutex

	provider      ai.Provider
	model         ai.Model
	thinkingLevel ai.ThinkingLevel
	systemPrompt  string
	tools         []agent.AgentTool
	session       *Session
	res           resources
	compaction    *CompactionConfig
	permission    *permission.Checker

	phase   Phase
	runCtx  context.Context
	runCancel context.CancelFunc

	steerQueue    []ai.UserMessage
	followUpQueue []ai.UserMessage

	bus *eventBus
}

// New constructs an AgentHarness.
func New(opts Options) *AgentHarness {
	if opts.ThinkingLevel == "" {
		opts.ThinkingLevel = ai.ThinkingOff
	}
	return &AgentHarness{
		provider:      opts.Provider,
		model:         opts.Model,
		thinkingLevel: opts.ThinkingLevel,
		systemPrompt:  opts.SystemPrompt,
		tools:         opts.Tools,
		session:       opts.Session,
		res:           resources{skills: opts.Skills, templates: opts.PromptTemplates},
		compaction:    opts.Compaction,
		permission:    opts.Permission,
		phase:         PhaseIdle,
		bus:           &eventBus{},
	}
}

// Phase returns the current run phase.
func (h *AgentHarness) Phase() Phase {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.phase
}

// SetModel changes the active model (idle only).
func (h *AgentHarness) SetModel(m ai.Model) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.phase != PhaseIdle {
		return fmt.Errorf("cannot set model while %s", h.phase)
	}
	h.model = m
	return h.appendEntry(ModelChangeEntry{
		EntryBase: EntryBase{Type: EntryModelChange, ID: h.session.Storage().CreateEntryID(), ParentID: h.session.GetLeafID(), Timestamp: nowISO()},
		Provider:  m.Provider,
		ModelID:   m.ID,
	})
}

// SetThinkingLevel changes the active thinking level (idle only).
func (h *AgentHarness) SetThinkingLevel(level ai.ThinkingLevel) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.phase != PhaseIdle {
		return fmt.Errorf("cannot set thinking level while %s", h.phase)
	}
	h.thinkingLevel = level
	return h.appendEntry(ThinkingLevelChangeEntry{
		EntryBase:      EntryBase{Type: EntryThinkingLevelChange, ID: h.session.Storage().CreateEntryID(), ParentID: h.session.GetLeafID(), Timestamp: nowISO()},
		ThinkingLevel:  string(level),
	})
}

// SetPermission replaces the permission checker at runtime (e.g. to wire a TUI
// asker after the harness is constructed). Safe to call before the first turn.
func (h *AgentHarness) SetPermission(c *permission.Checker) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.permission = c
}

// Permission returns the active checker (may be nil).
func (h *AgentHarness) Permission() *permission.Checker {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.permission
}

// Subscribe registers an event handler. Returns an unsubscribe function.
func (h *AgentHarness) Subscribe(handler EventHandler) func() {
	return h.bus.subscribe(handler)
}

func (h *AgentHarness) emit(ev agent.AgentEvent) {
	h.bus.emit(HarnessEvent{Agent: ev, Phase: h.phase})
}

// Prompt runs the agent loop with a new user message. Blocks until the run
// completes. Returns the new messages produced.
func (h *AgentHarness) Prompt(ctx context.Context, text string) ([]agent.AgentMessage, error) {
	h.mu.Lock()
	if h.phase != PhaseIdle {
		h.mu.Unlock()
		return nil, fmt.Errorf("harness is busy (%s)", h.phase)
	}
	h.phase = PhaseTurn
	runCtx, cancel := context.WithCancel(ctx)
	h.runCtx = runCtx
	h.runCancel = cancel
	h.mu.Unlock()

	defer func() {
		cancel()
		h.mu.Lock()
		h.phase = PhaseIdle
		h.runCtx = nil
		h.runCancel = nil
		h.mu.Unlock()
	}()

	// Rebuild context from session, then add the new prompt.
	sessCtx := h.session.BuildContext()

	// Automatic compaction before we start the turn, if configured.
	if err := h.compactIfNeeded(runCtx, sessCtx); err != nil {
		cancel()
		h.mu.Lock()
		h.phase = PhaseIdle
		h.runCtx = nil
		h.runCancel = nil
		h.mu.Unlock()
		return nil, fmt.Errorf("compaction failed: %w", err)
	}
	// Rebuild after compaction so the agent loop sees the compacted context.
	sessCtx = h.session.BuildContext()

	promptMsg := ai.UserMessage{Content: text, Timestamp: ai.Now()}
	// The prompt is persisted via the emit path: RunAgentLoop emits a
	// MessageEndEvent for each prompt (agent-loop.go), and persistFromEvent
	// appends user messages to the session. Persisting through one
	// authoritative path (rather than a separate explicit append here) keeps
	// the message ordering consistent: prompt becomes the leaf before any
	// assistant/tool-result message is appended.

	agentCtx := agent.AgentContext{
		SystemPrompt: h.effectiveSystemPrompt(),
		Tools:        h.tools,
		Messages:     toAgentMessages(sessCtx.Messages),
	}
	prompts := []agent.AgentMessage{promptMsg}

	builder := agent.NewLoopConfig(h.model).
		WithThinking(h.thinkingLevel).
		WithConvertToLlm(agent.DefaultConvertToLlm).
		WithSteering(h.drainSteer).
		WithFollowUp(h.drainFollowUp)
	if h.permission != nil {
		perm := h.permission
		beforeFn := func(ctx context.Context, c agent.BeforeToolCallContext) (*agent.BeforeToolCallResult, error) {
			args := permission.CheckArgs{
				ToolName: c.ToolCall.Name,
				Path:     pathArg(c.Args),
				Command:  commandArg(c.Args),
			}
			decision, reason := perm.Check(ctx, args)
			if decision == permission.DecisionDeny {
				return &agent.BeforeToolCallResult{Block: true, Reason: reason}, nil
			}
			return nil, nil
		}
		builder = builder.WithPermission(beforeFn, nil)
	}
	config := builder.Build()

	emit := func(ev agent.AgentEvent) error {
		h.emit(ev)
		// Persist assistant and tool-result messages as they finalize.
		h.persistFromEvent(ev)
		return nil
	}

	newMessages, err := agent.RunAgentLoop(runCtx, prompts, agentCtx, config, h.provider, emit)
	if err != nil {
		return newMessages, err
	}
	return newMessages, nil
}

// Skill invokes a named skill by injecting its content as the prompt text,
// optionally appended with additional user instructions.
func (h *AgentHarness) Skill(ctx context.Context, name string, additionalInstructions string) ([]agent.AgentMessage, error) {
	skill, ok := h.res.findSkill(name)
	if !ok {
		return nil, fmt.Errorf("unknown skill: %s", name)
	}
	return h.Prompt(ctx, FormatSkillInvocation(skill, additionalInstructions))
}

// PromptFromTemplate invokes a named prompt template with positional arguments.
func (h *AgentHarness) PromptFromTemplate(ctx context.Context, name string, args []string) ([]agent.AgentMessage, error) {
	t, ok := h.res.findTemplate(name)
	if !ok {
		return nil, fmt.Errorf("unknown prompt template: %s", name)
	}
	return h.Prompt(ctx, FormatPromptTemplateInvocation(t, args))
}

// effectiveSystemPrompt returns the configured system prompt with a list of
// available skills appended, delegated to the resources component.
func (h *AgentHarness) effectiveSystemPrompt() string {
	return h.res.effectiveSystemPrompt(h.systemPrompt)
}


// Steer enqueues a steering message to inject mid-run.
func (h *AgentHarness) Steer(text string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.phase == PhaseIdle {
		return fmt.Errorf("cannot steer while idle")
	}
	h.steerQueue = append(h.steerQueue, ai.UserMessage{Content: text, Timestamp: ai.Now()})
	return nil
}

// FollowUp enqueues a follow-up message processed after the agent would stop.
func (h *AgentHarness) FollowUp(text string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.phase == PhaseIdle {
		return fmt.Errorf("cannot follow up while idle")
	}
	h.followUpQueue = append(h.followUpQueue, ai.UserMessage{Content: text, Timestamp: ai.Now()})
	return nil
}

// Abort cancels the current run.
func (h *AgentHarness) Abort() {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.runCancel != nil {
		h.runCancel()
	}
}

// drainSteer returns queued steering messages (called by the loop between turns).
func (h *AgentHarness) drainSteer() []agent.AgentMessage {
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.steerQueue) == 0 {
		return nil
	}
	queue := h.steerQueue
	h.steerQueue = nil
	out := make([]agent.AgentMessage, 0, len(queue))
	for _, m := range queue {
		out = append(out, m)
	}
	return out
}

func (h *AgentHarness) drainFollowUp() []agent.AgentMessage {
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.followUpQueue) == 0 {
		return nil
	}
	queue := h.followUpQueue
	h.followUpQueue = nil
	out := make([]agent.AgentMessage, 0, len(queue))
	for _, m := range queue {
		out = append(out, m)
	}
	return out
}

// persistFromEvent appends finalized messages to the session.
func (h *AgentHarness) persistFromEvent(ev agent.AgentEvent) {
	switch e := ev.(type) {
	case agent.MessageEndEvent:
		if m, ok := e.Message.(ai.Message); ok {
			switch msg := m.(type) {
			case ai.AssistantMessage, ai.ToolResultMessage, ai.UserMessage:
				_ = h.appendMessage(msg)
			}
		}
	}
}

// appendMessage appends a message entry and advances the leaf.
func (h *AgentHarness) appendMessage(msg ai.Message) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.appendMessageLocked(msg)
}

func (h *AgentHarness) appendMessageLocked(msg ai.Message) error {
	return h.appendEntry(MessageEntry{
		EntryBase: EntryBase{
			Type:      EntryMessage,
			ID:        h.session.Storage().CreateEntryID(),
			ParentID:  h.session.GetLeafID(),
			Timestamp: nowISO(),
		},
		Message: msg,
	})
}

func (h *AgentHarness) appendEntry(e Entry) error {
	if err := h.session.Storage().AppendEntry(e); err != nil {
		return err
	}
	leaf := e.Base().ID
	if e.entryType() != EntryLeaf {
		return h.session.Storage().SetLeafID(&leaf)
	}
	return nil
}

// Session returns the underlying session.
func (h *AgentHarness) Session() *Session { return h.session }

func toAgentMessages(msgs []agent.AgentMessage) []agent.AgentMessage {
	return msgs
}

// pathArg extracts a file path from validated tool args (read/write/edit/grep/
// glob all use a "path" field). Returns "" for tools without a path.
func pathArg(args any) string {
	m, ok := args.(map[string]any)
	if !ok {
		return ""
	}
	if s, ok := m["path"].(string); ok {
		return s
	}
	return ""
}

// commandArg extracts the bash command string from validated tool args.
func commandArg(args any) string {
	m, ok := args.(map[string]any)
	if !ok {
		return ""
	}
	if s, ok := m["command"].(string); ok {
		return s
	}
	return ""
}

// compactIfNeeded runs automatic compaction when the session context exceeds
// configured limits. It summarizes the oldest messages, writes a CompactionEntry,
// and updates the session leaf so subsequent context rebuilds drop the old
// messages.
func (h *AgentHarness) compactIfNeeded(ctx context.Context, sessCtx SessionContext) error {
	cfg := h.compaction
	if cfg == nil || !cfg.Enabled {
		return nil
	}

	branch := h.session.GetBranch(nil)
	messageEntries := make([]MessageEntry, 0, len(branch))
	for _, e := range branch {
		if m, ok := e.(MessageEntry); ok {
			messageEntries = append(messageEntries, m)
		}
	}

	tokens := estimateTokens(sessCtx.Messages)
	maxMsgs := cfg.MaxMessages
	if maxMsgs == 0 {
		maxMsgs = 50
	}
	maxToks := cfg.MaxTokens
	if maxToks == 0 {
		// Sensible default: roughly 80% of a 128k context window, or 4000 if
		// the active model has no window configured.
		if h.model.ContextWindow > 0 {
			maxToks = h.model.ContextWindow * 8 / 10
		} else {
			maxToks = 4000
		}
	}
	keep := cfg.KeepMessages
	if keep < 2 {
		keep = 10
	}

	needsCompaction := len(messageEntries) > maxMsgs || tokens > maxToks
	if !needsCompaction {
		return nil
	}
	if len(messageEntries) <= keep {
		// Not enough history to drop anything meaningful.
		return nil
	}

	// We will drop messageEntries[0 : len-kept] and keep the rest.
	// The first kept message is messageEntries[len-kept].
	firstKept := messageEntries[len(messageEntries)-keep]
	toSummarize := messageEntries[:len(messageEntries)-keep]

	summary, err := h.summarizeMessages(ctx, toSummarize, tokens)
	if err != nil {
		return err
	}

	compaction := CompactionEntry{
		EntryBase: EntryBase{
			Type:      EntryCompaction,
			ID:        h.session.Storage().CreateEntryID(),
			ParentID:  h.session.GetLeafID(),
			Timestamp: nowISO(),
		},
		Summary:          summary,
		FirstKeptEntryID: firstKept.ID,
		TokensBefore:     tokens,
	}
	if err := h.session.Storage().AppendEntry(compaction); err != nil {
		return err
	}
	leaf := compaction.ID
	return h.session.Storage().SetLeafID(&leaf)
}

// summarizeMessages asks the LLM to summarize the provided messages so they can
// be compacted away.
func (h *AgentHarness) summarizeMessages(ctx context.Context, entries []MessageEntry, tokensBefore int) (string, error) {
	var sb strings.Builder
	sb.WriteString("Summarize the following conversation history so the user and assistant can continue without losing important context. Be concise but retain facts, decisions, open tasks, and file paths.\n\n")
	for _, e := range entries {
		switch m := e.Message.(type) {
		case ai.UserMessage:
			sb.WriteString("User: ")
			writeMessageContent(&sb, m.Content)
			sb.WriteString("\n")
		case ai.AssistantMessage:
			sb.WriteString("Assistant: ")
			for _, c := range m.Content {
				if t, ok := c.(ai.TextContent); ok {
					sb.WriteString(t.Text)
				}
			}
			sb.WriteString("\n")
		case ai.ToolResultMessage:
			sb.WriteString(fmt.Sprintf("Tool result (%s): ", m.ToolName))
			for _, c := range m.Content {
				if t, ok := c.(ai.TextContent); ok {
					sb.WriteString(t.Text)
				}
			}
			sb.WriteString("\n")
		}
	}
	sb.WriteString("\nSummary:")

	prompt := sb.String()
	model := h.model
	if h.compaction != nil && h.compaction.SummaryModel != nil {
		model = *h.compaction.SummaryModel
	}

	llmCtx := ai.Context{
		SystemPrompt: "You are a helpful assistant that summarizes conversation history.",
		Messages: []ai.Message{ai.UserMessage{Content: prompt, Timestamp: ai.Now()}},
	}
	stream := h.provider.StreamSimple(ctx, model, llmCtx, &ai.SimpleStreamOptions{
		StreamOptions: ai.StreamOptions{MaxTokens: 1024},
	})

	var textParts []string
	for ev := range stream.Range {
		switch e := ev.(type) {
		case ai.TextDeltaEvent:
			textParts = append(textParts, e.Delta)
		case ai.ErrorEvent:
			if final, _ := ai.TerminalMessage(e); final.ErrorMessage != "" {
				return "", fmt.Errorf("summary stream error: %s", final.ErrorMessage)
			}
		}
	}
	summary := strings.TrimSpace(strings.Join(textParts, ""))
	if summary == "" {
		// Fall back to a plain placeholder so compaction still works if the
		// model returns nothing (e.g. in tests with a mock that emits no text).
		summary = fmt.Sprintf("[compacted %d messages, ~%d tokens]", len(entries), tokensBefore)
	}
	return summary, nil
}

// writeMessageContent writes string or []ContentBlock content to a builder.
func writeMessageContent(sb *strings.Builder, content any) {
	switch v := content.(type) {
	case string:
		sb.WriteString(v)
	case []ai.ContentBlock:
		for _, c := range v {
			if t, ok := c.(ai.TextContent); ok {
				sb.WriteString(t.Text)
			}
		}
	}
}

// estimateTokens returns a rough token estimate for the visible messages. It
// uses a simple characters-per-token heuristic because the project does not
// bundle a tokenizer.
func estimateTokens(msgs []agent.AgentMessage) int {
	const charsPerToken = 4
	total := 0
	for _, m := range msgs {
		switch msg := m.(type) {
		case ai.UserMessage:
			total += estimateContentTokens(msg.Content) + 4 // message overhead
		case ai.AssistantMessage:
			for _, c := range msg.Content {
				if t, ok := c.(ai.TextContent); ok {
					total += len(t.Text)/charsPerToken + 1
				}
			}
			total += 4
		case ai.ToolResultMessage:
			for _, c := range msg.Content {
				if t, ok := c.(ai.TextContent); ok {
					total += len(t.Text)/charsPerToken + 1
				}
			}
			total += 4
		}
	}
	return total
}

func estimateContentTokens(content any) int {
	const charsPerToken = 4
	switch v := content.(type) {
	case string:
		return len(v)/charsPerToken + 4
	case []ai.ContentBlock:
		n := 4
		for _, c := range v {
			if t, ok := c.(ai.TextContent); ok {
				n += len(t.Text)/charsPerToken + 1
			}
		}
		return n
	}
	return 4
}
