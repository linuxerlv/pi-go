package harness

import (
	"context"
	"fmt"
	"sync"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

// Phase is the harness run state.
type Phase string

const (
	PhaseIdle           Phase = "idle"
	PhaseTurn           Phase = "turn"
	PhaseCompaction     Phase = "compaction"
	PhaseBranchSummary  Phase = "branch_summary"
	PhaseRetry          Phase = "retry"
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

	phase   Phase
	runCtx  context.Context
	runCancel context.CancelFunc

	steerQueue    []ai.UserMessage
	followUpQueue []ai.UserMessage

	handlers []EventHandler
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
		phase:         PhaseIdle,
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

// Subscribe registers an event handler. Returns an unsubscribe function.
func (h *AgentHarness) Subscribe(handler EventHandler) func() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.handlers = append(h.handlers, handler)
	idx := len(h.handlers) - 1
	return func() {
		h.mu.Lock()
		defer h.mu.Unlock()
		if idx < len(h.handlers) {
			h.handlers[idx] = nil
		}
	}
}

func (h *AgentHarness) emit(ev agent.AgentEvent) {
	handlers := h.handlersSnapshot()
	he := HarnessEvent{Agent: ev, Phase: h.phase}
	for _, handler := range handlers {
		if handler != nil {
			_ = handler(he)
		}
	}
}

func (h *AgentHarness) handlersSnapshot() []EventHandler {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]EventHandler, len(h.handlers))
	copy(out, h.handlers)
	return out
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
	promptMsg := ai.UserMessage{Content: text, Timestamp: ai.Now()}
	if err := h.appendMessageLocked(promptMsg); err != nil {
		return nil, err
	}

	agentCtx := agent.AgentContext{
		SystemPrompt: h.systemPrompt,
		Tools:        h.tools,
		Messages:     toAgentMessages(sessCtx.Messages),
	}
	prompts := []agent.AgentMessage{promptMsg}

	config := agent.AgentLoopConfig{
		Model:          h.model,
		ThinkingLevel:  h.thinkingLevel,
		ConvertToLlm:   agent.DefaultConvertToLlm,
		GetSteeringMessages: h.drainSteer,
		GetFollowUpMessages: h.drainFollowUp,
	}

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
			case ai.AssistantMessage, ai.ToolResultMessage:
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
