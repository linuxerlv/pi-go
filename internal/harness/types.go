// Package harness wraps the agent loop with stateful session management,
// persistence, compaction, and a steer/follow-up queue system. It is a Go port
// of @earendil-works/pi-agent-core's harness/ layer.
package harness

import (
	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

// EntryType identifies a session tree entry variant.
type EntryType string

const (
	EntryMessage             EntryType = "message"
	EntryThinkingLevelChange EntryType = "thinking_level_change"
	EntryModelChange         EntryType = "model_change"
	EntryActiveToolsChange   EntryType = "active_tools_change"
	EntryCompaction          EntryType = "compaction"
	EntryBranchSummary       EntryType = "branch_summary"
	EntryCustomMessage       EntryType = "custom_message"
	EntryLabel               EntryType = "label"
	EntryLeaf                EntryType = "leaf"
	EntrySessionInfo         EntryType = "session_info"
)

// EntryBase is the common shape of every session tree entry. Entries form a
// tree via ParentID; the active branch is recorded by a Leaf entry.
type EntryBase struct {
	Type      EntryType `json:"type"`
	ID        string    `json:"id"`
	ParentID  *string   `json:"parentId,omitempty"`
	Timestamp string    `json:"timestamp"`
}

// MessageEntry records a conversation message (user/assistant/toolResult, or a
// custom app message).
type MessageEntry struct {
	EntryBase
	Message agent.AgentMessage `json:"message"`
}

// ThinkingLevelChangeEntry records a thinking-level change.
type ThinkingLevelChangeEntry struct {
	EntryBase
	ThinkingLevel string `json:"thinkingLevel"`
}

// ModelChangeEntry records a model switch.
type ModelChangeEntry struct {
	EntryBase
	Provider string `json:"provider"`
	ModelID  string `json:"modelId"`
}

// ActiveToolsChangeEntry records which tools are enabled.
type ActiveToolsChangeEntry struct {
	EntryBase
	ActiveToolNames []string `json:"activeToolNames"`
}

// CompactionEntry records a context compaction: a summary plus the first entry
// id kept after the summary (earlier entries are dropped from context).
type CompactionEntry struct {
	EntryBase
	Summary          string `json:"summary"`
	FirstKeptEntryID string `json:"firstKeptEntryId"`
	TokensBefore     int    `json:"tokensBefore"`
	FromHook         bool   `json:"fromHook,omitempty"`
}

// BranchSummaryEntry records a summary of a forked branch, injected when
// continuing from a non-leaf entry.
type BranchSummaryEntry struct {
	EntryBase
	FromID  string `json:"fromId"`
	Summary string `json:"summary"`
	FromHook bool  `json:"fromHook,omitempty"`
}

// LeafEntry records the active session-tree leaf (the current branch tip).
type LeafEntry struct {
	EntryBase
	TargetID *string `json:"targetId"`
}

// LabelEntry attaches a human-readable label to a target entry.
type LabelEntry struct {
	EntryBase
	TargetID string  `json:"targetId"`
	Label    *string `json:"label"`
}

// SessionContext is the rebuild result from a branch: the visible messages
// plus the active model/thinking/tools state.
type SessionContext struct {
	Messages        []agent.AgentMessage
	ThinkingLevel   ai.ThinkingLevel
	Model           *ModelRef
	ActiveToolNames []string
}

// ModelRef is a (provider, modelId) reference reconstructed from history.
type ModelRef struct {
	Provider string
	ModelID  string
}

// CompactionConfig configures automatic context compaction for an
// AgentHarness. When the visible conversation exceeds the configured limits,
// the harness asks the model to summarize the oldest messages and stores a
// CompactionEntry so future context rebuilds drop those messages.
type CompactionConfig struct {
	// Enabled turns automatic compaction on. Manual compaction can still be
	// performed by writing CompactionEntry directly into session storage.
	Enabled bool
	// MaxMessages triggers compaction when the number of visible messages
	// exceeds this value. Zero disables the message-count trigger.
	MaxMessages int
	// MaxTokens triggers compaction when the estimated token count exceeds
	// this value. Zero disables the token trigger.
	MaxTokens int
	// KeepMessages is how many of the most recent messages to retain after
	// compaction. Must be >= 2 to keep a user/assistant pair. Defaults to 10.
	KeepMessages int
	// SummaryModel is the model used to generate the summary. If nil, the
	// harness's active model is used.
	SummaryModel *ai.Model
	// SummaryPrompt overrides the default compaction summary prompt.
	SummaryPrompt string
}

// SessionMetadata is the per-session metadata.
type SessionMetadata struct {
	ID        string `json:"id"`
	CreatedAt string `json:"createdAt"`
}
