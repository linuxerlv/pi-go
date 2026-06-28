// Package tui provides the pi-go interactive terminal UI built on bubbletea.
// It renders a scrolling conversation (streaming assistant text, tool blocks,
// status bar) and a text-input line, and forwards agent events from the
// harness into the bubbletea event loop.
package tui

import (
	"context"
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

// style definitions.
var (
	userStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("63")).Bold(true)
	asstStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("39"))
	toolStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("214"))
	errStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("203"))
	dimStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	statusStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("245")).Background(lipgloss.Color("236"))
	promptStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("39")).Bold(true)
	permPromptStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("221")).Bold(true)
)

// agentEventMsg wraps an agent event delivered to the bubbletea loop.
type agentEventMsg struct{ event agent.AgentEvent }

// permRequestMsg is sent when the permission checker needs a decision. The
// TUI shows a prompt and waits for the user to press y/n/a.
type permRequestMsg struct {
	prompt  string
	replyCh chan string
}

// runDoneMsg is sent when the harness run completes.
type runDoneMsg struct{ err error }

// Model is the bubbletea model for the pi-go TUI.
type Model struct {
	harness    HarnessRunner
	textInput  textinput.Model
	history    []historyLine
	status     string
	modelName  string
	width      int
	height     int
	running    bool // a prompt is in flight
	quit       bool

	// permission modal state
	permPrompt  string
	permReplyCh chan string
	permActive  bool

	history_input []string // user input history
	histIdx       int
	slashHandler  func(line string) (handled bool, output string)
}

// historyLine is one rendered line of the conversation.
type historyLine struct {
	text  string
	style lipgloss.Style
}

// HarnessRunner is the harness interface the TUI depends on (decoupled from
// the concrete *harness.AgentHarness to avoid an import cycle). The CLI
// supplies an adapter wrapping *harness.AgentHarness.
type HarnessRunner interface {
	// Subscribe registers an agent-event handler. Returns an unsubscribe func.
	Subscribe(handler func(agent.AgentEvent)) func()
	// Prompt runs one user turn and blocks until it completes.
	Prompt(ctx context.Context, text string) ([]agent.AgentMessage, error)
	// Phase returns the current harness phase name.
	Phase() string
}

// NewModel constructs the TUI model.
func NewModel(h HarnessRunner, modelName string) Model {
	ti := textinput.New()
	ti.Placeholder = "type a prompt, or /help for commands"
	ti.Prompt = "▶ "
	ti.PromptStyle = promptStyle
	ti.CharLimit = 0
	ti.Focus()

	return Model{
		harness:   h,
		textInput: ti,
		modelName: modelName,
		status:    "idle",
	}
}

// Init starts the bubbletea program.
func (m Model) Init() tea.Cmd {
	return nil
}

// Update handles messages.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.textInput.Width = max(msg.Width-4, 20)
		return m, nil

	case tea.KeyMsg:
		return m.handleKey(msg)

	case agentEventMsg:
		m.applyEvent(msg.event)
		return m, nil

	case permRequestMsg:
		m.permActive = true
		m.permPrompt = msg.prompt
		m.permReplyCh = msg.replyCh
		return m, nil

	case runDoneMsg:
		m.running = false
		m.status = "idle"
		if msg.err != nil {
			m.appendLine(errStyle.Render("[error: "+msg.err.Error()+"]"))
		}
		return m, nil
	}

	var cmd tea.Cmd
	m.textInput, cmd = m.textInput.Update(msg)
	return m, cmd
}

func (m Model) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	// Permission modal takes priority.
	if m.permActive {
		switch msg.String() {
		case "y", "Y":
			m.answerPerm("allow")
		case "a", "A":
			m.answerPerm("allow-always")
		case "n", "N", "esc":
			m.answerPerm("deny")
		}
		return m, nil
	}

	switch msg.String() {
	case "ctrl+c", "ctrl+d":
		m.quit = true
		return m, tea.Quit
	case "enter":
		if m.running {
			return m, nil
		}
		text := strings.TrimSpace(m.textInput.Value())
		if text == "" {
			return m, nil
		}
		m.textInput.SetValue("")
		m.history_input = append(m.history_input, text)
		m.histIdx = len(m.history_input)
		m.appendLine(userStyle.Render("you: ") + text)
		if strings.HasPrefix(text, "/") {
			if m.slashHandler != nil {
				handled, output := m.slashHandler(text)
				if output != "" {
					m.appendLine(dimStyle.Render(output))
				}
				if handled {
					return m, nil
				}
			} else {
				m.appendLine(dimStyle.Render("(no slash handler wired)"))
			}
			return m, nil
		}
		m.running = true
		m.status = "running"
		return m, m.startRun(text)
	case "up":
		if m.histIdx > 0 {
			m.histIdx--
			m.textInput.SetValue(m.history_input[m.histIdx])
		}
	case "down":
		if m.histIdx < len(m.history_input) {
			m.histIdx++
			if m.histIdx < len(m.history_input) {
				m.textInput.SetValue(m.history_input[m.histIdx])
			} else {
				m.textInput.SetValue("")
			}
		}
	case "ctrl+l":
		m.history = nil
	}

	var cmd tea.Cmd
	m.textInput, cmd = m.textInput.Update(msg)
	return m, cmd
}

func (m *Model) answerPerm(answer string) {
	ch := m.permReplyCh
	m.permActive = false
	m.permPrompt = ""
	m.permReplyCh = nil
	if ch != nil {
		ch <- answer
	}
}

// startRun launches the harness prompt in a goroutine and returns a tea.Cmd
// that waits for completion, forwarding agent events into the loop.
func (m Model) startRun(text string) tea.Cmd {
	h := m.harness
	return func() tea.Msg {
		// Events are delivered via the subscription set up in NewProgram.
		_, err := h.Prompt(context.Background(), text)
		return runDoneMsg{err: err}
	}
}

// applyEvent appends rendered lines from an agent event.
func (m *Model) applyEvent(ev agent.AgentEvent) {
	switch e := ev.(type) {
	case agent.AgentStartEvent:
		m.status = "running"
	case agent.TurnStartEvent:
		// nothing
	case agent.MessageStartEvent:
		if _, ok := e.Message.(ai.AssistantMessage); ok {
			m.appendLine(asstStyle.Render("assistant:"))
		}
	case agent.MessageUpdateEvent:
		if d, ok := e.AssistantMessageEvent.(ai.TextDeltaEvent); ok {
			m.appendToLast(d.Delta)
		}
	case agent.ToolExecutionStartEvent:
		m.appendLine(toolStyle.Render(fmt.Sprintf("▶ %s", e.ToolName)))
	case agent.ToolExecutionEndEvent:
		marker := "✓"
		style := toolStyle
		if e.IsError {
			marker = "✗"
			style = errStyle
		}
		m.appendLine(style.Render(fmt.Sprintf("%s %s", marker, e.ToolName)))
	case agent.AgentEndEvent:
		m.status = "done"
	}
}

func (m *Model) appendLine(s string) {
	m.history = append(m.history, historyLine{text: s})
}

func (m *Model) appendToLast(s string) {
	if len(m.history) == 0 {
		m.appendLine(s)
		return
	}
	m.history[len(m.history)-1].text += s
}

// View renders the TUI.
func (m Model) View() string {
	if m.width == 0 {
		return "starting…"
	}
	var b strings.Builder
	// Conversation region (leave room for input + status).
	convHeight := m.height - 3
	if convHeight < 3 {
		convHeight = 3
	}
	start := 0
	if len(m.history) > convHeight {
		start = len(m.history) - convHeight
	}
	for i := start; i < len(m.history); i++ {
		b.WriteString(m.history[i].text)
		b.WriteString("\n")
	}
	// Pad to fill.
	rendered := b.String()
	lines := strings.Count(rendered, "\n")
	for i := lines; i < convHeight; i++ {
		rendered += "\n"
	}

	// Permission modal (overlays the input line).
	if m.permActive {
		return rendered + "\n" + permPromptStyle.Render("permission: "+m.permPrompt+" [y]es/[n]o/[a]lways")
	}

	// Status bar + input.
	status := statusStyle.Render(fmt.Sprintf(" %s | %s | %s ", m.modelName, m.status, m.phaseLabel()))
	return rendered + "\n" + status + "\n" + m.textInput.View()
}

func (m Model) phaseLabel() string {
	if m.harness == nil {
		return ""
	}
	return m.harness.Phase()
}

// SendEvent delivers an agent event into the bubbletea loop. Call this from the
// harness subscription handler.
func (m *Model) SendEvent(p *tea.Program, ev agent.AgentEvent) {
	p.Send(agentEventMsg{event: ev})
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
