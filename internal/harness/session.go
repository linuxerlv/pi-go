package harness

import (
	"sync"
	"time"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

// Entry is a session tree entry. Concrete types: MessageEntry,
// ThinkingLevelChangeEntry, ModelChangeEntry, ActiveToolsChangeEntry,
// CompactionEntry, BranchSummaryEntry, LeafEntry, LabelEntry.
type Entry interface {
	Base() EntryBase
	entryType() EntryType
}

func (MessageEntry) entryType() EntryType             { return EntryMessage }
func (ThinkingLevelChangeEntry) entryType() EntryType { return EntryThinkingLevelChange }
func (ModelChangeEntry) entryType() EntryType         { return EntryModelChange }
func (ActiveToolsChangeEntry) entryType() EntryType   { return EntryActiveToolsChange }
func (CompactionEntry) entryType() EntryType          { return EntryCompaction }
func (BranchSummaryEntry) entryType() EntryType       { return EntryBranchSummary }
func (LeafEntry) entryType() EntryType                { return EntryLeaf }
func (LabelEntry) entryType() EntryType               { return EntryLabel }

func (e MessageEntry) Base() EntryBase             { return e.EntryBase }
func (e ThinkingLevelChangeEntry) Base() EntryBase { return e.EntryBase }
func (e ModelChangeEntry) Base() EntryBase         { return e.EntryBase }
func (e ActiveToolsChangeEntry) Base() EntryBase   { return e.EntryBase }
func (e CompactionEntry) Base() EntryBase          { return e.EntryBase }
func (e BranchSummaryEntry) Base() EntryBase       { return e.EntryBase }
func (e LeafEntry) Base() EntryBase                { return e.EntryBase }
func (e LabelEntry) Base() EntryBase               { return e.EntryBase }

// SessionStorage is the persistence layer for a session's entry tree.
type SessionStorage interface {
	Metadata() SessionMetadata
	GetLeafID() *string
	SetLeafID(leafID *string) error
	CreateEntryID() string
	AppendEntry(entry Entry) error
	GetEntry(id string) (Entry, bool)
	FindEntries(t EntryType) []Entry
	GetPathToRoot(leafID *string) []Entry
	AllEntries() []Entry
}

// Session is a stateful conversation branch. It wraps a SessionStorage and
// rebuilds context from the entry tree.
type Session struct {
	storage SessionStorage
}

// NewSession wraps a storage.
func NewSession(storage SessionStorage) *Session { return &Session{storage: storage} }

// Storage returns the underlying storage.
func (s *Session) Storage() SessionStorage { return s.storage }

// Metadata returns session metadata.
func (s *Session) Metadata() SessionMetadata { return s.storage.Metadata() }

// GetLeafID returns the active leaf id.
func (s *Session) GetLeafID() *string { return s.storage.GetLeafID() }

// GetEntry returns an entry by id.
func (s *Session) GetEntry(id string) (Entry, bool) { return s.storage.GetEntry(id) }

// GetBranch returns the entry path from root to leaf (or to fromID).
func (s *Session) GetBranch(fromID *string) []Entry {
	leaf := fromID
	if leaf == nil {
		leaf = s.storage.GetLeafID()
	}
	return s.storage.GetPathToRoot(leaf)
}

// BuildContext reconstructs the SessionContext from the current branch. It
// applies compaction: a compaction entry's summary is injected, and only
// entries from FirstKeptEntryID onward (plus entries after the compaction) are
// visible. Mirrors pi-agent-core's buildSessionContext.
func (s *Session) BuildContext() SessionContext {
	return buildSessionContext(s.GetBranch(nil))
}

// AppendMessage appends a message entry.
func (s *Session) AppendMessage(msg agent.AgentMessage) (MessageEntry, error) {
	e := MessageEntry{
		EntryBase: EntryBase{
			Type:      EntryMessage,
			ID:        s.storage.CreateEntryID(),
			ParentID:  s.storage.GetLeafID(),
			Timestamp: nowISO(),
		},
		Message: msg,
	}
	if err := s.storage.AppendEntry(e); err != nil {
		return MessageEntry{}, err
	}
	leaf := e.ID
	_ = s.storage.SetLeafID(&leaf)
	return e, nil
}

// buildSessionContext reconstructs messages and runtime state from a branch's
// entries, honoring compaction. Mirrors session.ts:buildSessionContext.
func buildSessionContext(path []Entry) SessionContext {
	ctx := SessionContext{ThinkingLevel: ai.ThinkingOff}

	var compaction *CompactionEntry
	for _, e := range path {
		switch v := e.(type) {
		case ThinkingLevelChangeEntry:
			ctx.ThinkingLevel = ai.ThinkingLevel(v.ThinkingLevel)
		case ModelChangeEntry:
			ctx.Model = &ModelRef{Provider: v.Provider, ModelID: v.ModelID}
		case MessageEntry:
			if m, ok := v.Message.(ai.AssistantMessage); ok {
				ctx.Model = &ModelRef{Provider: m.Provider, ModelID: m.Model}
			}
		case ActiveToolsChangeEntry:
			ctx.ActiveToolNames = append([]string(nil), v.ActiveToolNames...)
		case CompactionEntry:
			c := v
			compaction = &c
		}
	}

	appendMessage := func(e Entry) {
		switch v := e.(type) {
		case MessageEntry:
			ctx.Messages = append(ctx.Messages, v.Message)
		case BranchSummaryEntry:
			// Branch summaries are surfaced as a user message describing the
			// prior branch, so the model has continuity.
			ctx.Messages = append(ctx.Messages, ai.UserMessage{
				Content:   "[branch summary] " + v.Summary,
				Timestamp: ai.Now(),
			})
		}
	}

	if compaction != nil {
		// Inject compaction summary, then only entries from FirstKeptEntryID to
		// the compaction, then entries after the compaction.
		ctx.Messages = append(ctx.Messages, ai.UserMessage{
			Content:   "[compaction summary] " + compaction.Summary,
			Timestamp: ai.Now(),
		})
		compactionIdx := -1
		for i, e := range path {
			if c, ok := e.(CompactionEntry); ok && c.ID == compaction.ID {
				compactionIdx = i
				break
			}
		}
		foundFirstKept := false
		for i := 0; i < compactionIdx && i < len(path); i++ {
			if path[i].Base().ID == compaction.FirstKeptEntryID {
				foundFirstKept = true
			}
			if foundFirstKept {
				appendMessage(path[i])
			}
		}
		for i := compactionIdx + 1; i < len(path); i++ {
			appendMessage(path[i])
		}
	} else {
		for _, e := range path {
			appendMessage(e)
		}
	}
	return ctx
}

func nowISO() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// MemoryStorage is an in-memory SessionStorage. It is concurrency-safe and
// useful for tests and ephemeral sessions.
type MemoryStorage struct {
	mu      sync.Mutex
	meta    SessionMetadata
	entries map[string]Entry
	order   []string // entry ids in append order
	leafID  *string
	counter int
}

// NewMemoryStorage creates an in-memory storage with the given metadata.
func NewMemoryStorage(meta SessionMetadata) *MemoryStorage {
	return &MemoryStorage{meta: meta, entries: map[string]Entry{}}
}

func (m *MemoryStorage) Metadata() SessionMetadata { return m.meta }

func (m *MemoryStorage) GetLeafID() *string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.leafID == nil {
		return nil
	}
	cp := *m.leafID
	return &cp
}

func (m *MemoryStorage) SetLeafID(leafID *string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if leafID == nil {
		m.leafID = nil
		return nil
	}
	cp := *leafID
	m.leafID = &cp
	return nil
}

func (m *MemoryStorage) CreateEntryID() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.counter++
	// Simple monotonic id; sufficient for in-memory and tests.
	return fmtEntryID(m.meta.ID, m.counter)
}

func (m *MemoryStorage) AppendEntry(entry Entry) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	id := entry.Base().ID
	m.entries[id] = entry
	m.order = append(m.order, id)
	return nil
}

func (m *MemoryStorage) GetEntry(id string) (Entry, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	e, ok := m.entries[id]
	return e, ok
}

func (m *MemoryStorage) FindEntries(t EntryType) []Entry {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []Entry
	for _, id := range m.order {
		if e, ok := m.entries[id]; ok && e.entryType() == t {
			out = append(out, e)
		}
	}
	return out
}

func (m *MemoryStorage) AllEntries() []Entry {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Entry, 0, len(m.order))
	for _, id := range m.order {
		if e, ok := m.entries[id]; ok {
			out = append(out, e)
		}
	}
	return out
}

// GetPathToRoot walks from leafID up via ParentID and returns root..leaf order.
func (m *MemoryStorage) GetPathToRoot(leafID *string) []Entry {
	m.mu.Lock()
	defer m.mu.Unlock()
	var stack []Entry
	var current *string
	if leafID != nil {
		c := *leafID
		current = &c
	} else {
		current = m.leafID
	}
	for current != nil {
		e, ok := m.entries[*current]
		if !ok {
			break
		}
		stack = append(stack, e)
		pid := e.Base().ParentID
		if pid == nil {
			break
		}
		current = pid
	}
	// Reverse to root-first order.
	for i, j := 0, len(stack)-1; i < j; i, j = i+1, j-1 {
		stack[i], stack[j] = stack[j], stack[i]
	}
	return stack
}

func fmtEntryID(sessionID string, n int) string {
	return sessionID + "-e" + itoa(n)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
