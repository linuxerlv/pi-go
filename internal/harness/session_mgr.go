package harness

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// SessionManager lists, creates, and opens sessions stored as jsonl files in a
// directory. It is the data layer behind the /sessions, /resume, and /new
// slash commands.
type SessionManager struct {
	dir string
}

// NewSessionManager constructs a SessionManager rooted at dir (typically
// ".pi-go/sessions").
func NewSessionManager(dir string) *SessionManager {
	return &SessionManager{dir: dir}
}

// SessionInfo is a lightweight summary of a stored session.
type SessionInfo struct {
	ID         string
	CreatedAt  string
	MessageCount int
	Label      string
}

// List scans the session directory and returns summaries sorted by creation
// time descending (newest first). Missing directory returns an empty slice.
func (m *SessionManager) List() ([]SessionInfo, error) {
	entries, err := os.ReadDir(m.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var infos []SessionInfo
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		id := strings.TrimSuffix(e.Name(), ".jsonl")
		storage, err := NewJsonlStorage(m.dir, id)
		if err != nil {
			continue
		}
		sess := NewSession(storage)
		ctx := sess.BuildContext()
		info := SessionInfo{
			ID:           id,
			CreatedAt:    storage.Metadata().CreatedAt,
			MessageCount: len(ctx.Messages),
		}
		// Look for a label entry on the branch.
		for _, ent := range sess.Storage().FindEntries(EntryLabel) {
			if l, ok := ent.(LabelEntry); ok && l.Label != nil {
				info.Label = *l.Label
				break
			}
		}
		infos = append(infos, info)
	}
	sort.Slice(infos, func(i, j int) bool {
		return infos[i].CreatedAt > infos[j].CreatedAt
	})
	return infos, nil
}

// Open loads an existing session by id. Returns an error if it does not exist.
func (m *SessionManager) Open(id string) (*Session, error) {
	if _, err := os.Stat(filepath.Join(m.dir, id+".jsonl")); err != nil {
		return nil, err
	}
	storage, err := NewJsonlStorage(m.dir, id)
	if err != nil {
		return nil, err
	}
	return NewSession(storage), nil
}

// Create makes a new empty session with the given id. If id is empty, a
// timestamp-derived id is generated.
func (m *SessionManager) Create(id string) (*Session, error) {
	if id == "" {
		id = "sess-" + nowCompact()
	}
	storage, err := NewJsonlStorage(m.dir, id)
	if err != nil {
		return nil, err
	}
	return NewSession(storage), nil
}

// nowCompact returns a filesystem-safe timestamp for default session ids.
func nowCompact() string {
	// Reuse nowISO but strip punctuation; RFC3339Nano has ':' which is fine on
	// most filesystems but ugly. Replace with nothing for compactness.
	return strings.NewReplacer(":", "", "-", "", ".", "").Replace(nowISO())
}
