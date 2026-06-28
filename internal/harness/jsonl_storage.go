package harness

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// JsonlStorage is a file-backed SessionStorage. Each entry is one JSON line in
// <dir>/<sessionID>.jsonl. A sidecar <sessionID>.meta.json holds metadata and
// the current leaf id.
type JsonlStorage struct {
	mu      sync.Mutex
	dir     string
	meta    SessionMetadata
	entries map[string]Entry
	order   []string
	leafID  *string
	counter int
}

// jsonlMetaFile is the sidecar metadata shape.
type jsonlMetaFile struct {
	Session SessionMetadata `json:"session"`
	LeafID  *string         `json:"leafId,omitempty"`
}

// NewJsonlStorage loads (or creates) a jsonl session at dir/<sessionID>.jsonl.
func NewJsonlStorage(dir, sessionID string) (*JsonlStorage, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	s := &JsonlStorage{
		dir:     dir,
		meta:    SessionMetadata{ID: sessionID, CreatedAt: nowISO()},
		entries: map[string]Entry{},
	}
	if err := s.load(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *JsonlStorage) entryPath() string  { return filepath.Join(s.dir, s.meta.ID+".jsonl") }
func (s *JsonlStorage) metaPath() string   { return filepath.Join(s.dir, s.meta.ID+".meta.json") }

func (s *JsonlStorage) load() error {
	// Load meta sidecar if present.
	if b, err := os.ReadFile(s.metaPath()); err == nil {
		var mf jsonlMetaFile
		if json.Unmarshal(b, &mf) == nil {
			s.meta = mf.Session
			if mf.LeafID != nil {
				id := *mf.LeafID
				s.leafID = &id
			}
		}
	} else if !os.IsNotExist(err) {
		return err
	}

	f, err := os.Open(s.entryPath())
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		e, err := unmarshalEntry([]byte(line))
		if err != nil {
			return fmt.Errorf("jsonl parse: %w", err)
		}
		s.entries[e.Base().ID] = e
		s.order = append(s.order, e.Base().ID)
		// Bump counter past the highest entry number seen.
		if n := entryNumber(e.Base().ID); n > s.counter {
			s.counter = n
		}
	}
	// If no leaf was recorded (e.g. an imported jsonl without a meta sidecar),
	// default the leaf to the last appended entry so the branch is visible.
	if s.leafID == nil && len(s.order) > 0 {
		last := s.order[len(s.order)-1]
		s.leafID = &last
	}
	return scanner.Err()
}

// persistMeta writes the sidecar metadata file.
func (s *JsonlStorage) persistMeta() error {
	mf := jsonlMetaFile{Session: s.meta, LeafID: s.leafID}
	b, err := json.MarshalIndent(mf, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.metaPath(), b, 0o644)
}

// Metadata returns session metadata.
func (s *JsonlStorage) Metadata() SessionMetadata { return s.meta }

// GetLeafID returns the active leaf id.
func (s *JsonlStorage) GetLeafID() *string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.leafID == nil {
		return nil
	}
	cp := *s.leafID
	return &cp
}

// SetLeafID updates and persists the leaf id.
func (s *JsonlStorage) SetLeafID(leafID *string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if leafID == nil {
		s.leafID = nil
	} else {
		cp := *leafID
		s.leafID = &cp
	}
	return s.persistMeta()
}

// CreateEntryID returns a new monotonic entry id.
func (s *JsonlStorage) CreateEntryID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.counter++
	return fmtEntryID(s.meta.ID, s.counter)
}

// AppendEntry appends an entry to the jsonl file and in-memory index.
func (s *JsonlStorage) AppendEntry(entry Entry) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	raw, err := marshalEntry(entry)
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	f, err := os.OpenFile(s.entryPath(), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.Write(raw); err != nil {
		return err
	}
	id := entry.Base().ID
	s.entries[id] = entry
	s.order = append(s.order, id)
	return nil
}

// GetEntry returns an entry by id.
func (s *JsonlStorage) GetEntry(id string) (Entry, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.entries[id]
	return e, ok
}

// FindEntries returns entries of a given type in append order.
func (s *JsonlStorage) FindEntries(t EntryType) []Entry {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []Entry
	for _, id := range s.order {
		if e, ok := s.entries[id]; ok && e.entryType() == t {
			out = append(out, e)
		}
	}
	return out
}

// AllEntries returns all entries in append order.
func (s *JsonlStorage) AllEntries() []Entry {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Entry, 0, len(s.order))
	for _, id := range s.order {
		if e, ok := s.entries[id]; ok {
			out = append(out, e)
		}
	}
	return out
}

// GetPathToRoot walks from leafID up via ParentID, returning root-first order.
func (s *JsonlStorage) GetPathToRoot(leafID *string) []Entry {
	s.mu.Lock()
	defer s.mu.Unlock()
	var stack []Entry
	current := leafID
	if current == nil {
		current = s.leafID
	}
	for current != nil {
		e, ok := s.entries[*current]
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
	for i, j := 0, len(stack)-1; i < j; i, j = i+1, j-1 {
		stack[i], stack[j] = stack[j], stack[i]
	}
	return stack
}

// ---- entry (un)marshaling ----

// marshalEntry encodes an Entry as a tagged-union JSON object.
func marshalEntry(e Entry) ([]byte, error) {
	switch v := e.(type) {
	case MessageEntry:
		msgRaw, err := ai.MarshalMessage(v.Message)
		if err != nil {
			return nil, err
		}
		return json.Marshal(struct {
			EntryBase
			Message json.RawMessage `json:"message"`
		}{v.EntryBase, json.RawMessage(msgRaw)})
	case ThinkingLevelChangeEntry:
		return json.Marshal(v)
	case ModelChangeEntry:
		return json.Marshal(v)
	case ActiveToolsChangeEntry:
		return json.Marshal(v)
	case CompactionEntry:
		return json.Marshal(v)
	case BranchSummaryEntry:
		return json.Marshal(v)
	case LeafEntry:
		return json.Marshal(v)
	case LabelEntry:
		return json.Marshal(v)
	}
	return nil, fmt.Errorf("unsupported entry type: %T", e)
}

// unmarshalEntry decodes a tagged-union JSON object into the right Entry.
func unmarshalEntry(data []byte) (Entry, error) {
	var probe struct {
		Type EntryType `json:"type"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, err
	}
	switch probe.Type {
	case EntryMessage:
		var raw struct {
			EntryBase
			Message json.RawMessage `json:"message"`
		}
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, err
		}
		msg, err := ai.UnmarshalMessage(raw.Message)
		if err != nil {
			return nil, err
		}
		return MessageEntry{EntryBase: raw.EntryBase, Message: msg}, nil
	case EntryThinkingLevelChange:
		var e ThinkingLevelChangeEntry
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, err
		}
		return e, nil
	case EntryModelChange:
		var e ModelChangeEntry
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, err
		}
		return e, nil
	case EntryActiveToolsChange:
		var e ActiveToolsChangeEntry
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, err
		}
		return e, nil
	case EntryCompaction:
		var e CompactionEntry
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, err
		}
		return e, nil
	case EntryBranchSummary:
		var e BranchSummaryEntry
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, err
		}
		return e, nil
	case EntryLeaf:
		var e LeafEntry
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, err
		}
		return e, nil
	case EntryLabel:
		var e LabelEntry
		if err := json.Unmarshal(data, &e); err != nil {
			return nil, err
		}
		return e, nil
	}
	return nil, fmt.Errorf("unknown entry type: %q", probe.Type)
}

// entryNumber extracts the trailing integer from an id like "<sessionID>-e12".
func entryNumber(id string) int {
	idx := strings.LastIndex(id, "-e")
	if idx < 0 {
		return 0
	}
	n := 0
	for _, r := range id[idx+2:] {
		if r < '0' || r > '9' {
			return 0
		}
		n = n*10 + int(r-'0')
	}
	return n
}
