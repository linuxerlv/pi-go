package permission

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
)

// Store persists allow-always decisions and project trust on disk under
// ~/.config/pi-go/.
type Store struct {
	mu       sync.Mutex
	dir      string
	permPath string // permissions.json
	trust    map[string]bool // cwd-canonicalized -> trusted
	allow    []StoredRule
}

// StoredRule is a persisted allow-always rule.
type StoredRule struct {
	Tool       string `json:"tool"`
	Kind       string `json:"kind"` // "deny-arg" | "allow-arg" | "allow-path"
	ArgPattern string `json:"argPattern,omitempty"`
	PathPrefix string `json:"pathPrefix,omitempty"`
}

// trustFile is the on-disk trust.json shape.
type trustFile map[string]bool

// NewStore loads (or creates) a store at ~/.config/pi-go/. Returns nil (no
// error) if the config dir cannot be determined, disabling persistence.
func NewStore() *Store {
	dir := configDir()
	if dir == "" {
		return nil
	}
	dir = filepath.Join(dir, "pi-go")
	_ = os.MkdirAll(dir, 0o755)
	s := &Store{
		dir:      dir,
		permPath: filepath.Join(dir, "permissions.json"),
		trust:    trustFile{},
	}
	s.load()
	return s
}

// WithDir returns a store rooted at an explicit dir (for tests).
func WithDir(dir string) *Store {
	_ = os.MkdirAll(dir, 0o755)
	s := &Store{
		dir:      dir,
		permPath: filepath.Join(dir, "permissions.json"),
		trust:    trustFile{},
	}
	s.load()
	return s
}

func (s *Store) load() {
	if b, err := os.ReadFile(s.permPath); err == nil {
		var data struct {
			Trust map[string]bool `json:"trust"`
			Allow []StoredRule    `json:"allow"`
		}
		if json.Unmarshal(b, &data) == nil {
			if data.Trust != nil {
				s.trust = data.Trust
			}
			s.allow = data.Allow
		}
	}
}

func (s *Store) save() {
	data := struct {
		Trust map[string]bool `json:"trust"`
		Allow []StoredRule    `json:"allow"`
	}{Trust: s.trust, Allow: s.allow}
	b, _ := json.MarshalIndent(data, "", "  ")
	_ = os.WriteFile(s.permPath, b, 0o644)
}

// IsAllowedAlways reports whether a persisted rule allows this call.
func (s *Store) IsAllowedAlways(tool string, args CheckArgs) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, r := range s.allow {
		if r.Tool != tool {
			continue
		}
		if r.ArgPattern != "" {
			re, err := regexp.Compile(r.ArgPattern)
			if err == nil {
				target := args.Command
				if target == "" {
					target = args.Path
				}
				if re.MatchString(target) {
					return true
				}
			}
		}
		if r.PathPrefix != "" && args.Path != "" {
			abs, err := filepath.Abs(args.Path)
			if err == nil && strings.HasPrefix(abs, r.PathPrefix) {
				return true
			}
		}
	}
	return false
}

// AddAllowAlways persists an allow-always rule derived from the call args.
func (s *Store) AddAllowAlways(tool string, args CheckArgs) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := StoredRule{Tool: tool}
	if args.Command != "" {
		// Persist the exact command as a literal regex.
		r.ArgPattern = "^" + regexp.QuoteMeta(args.Command) + "$"
	} else if args.Path != "" {
		abs, err := filepath.Abs(args.Path)
		if err == nil {
			r.PathPrefix = abs
		}
	}
	s.allow = append(s.allow, r)
	s.save()
	return nil
}

// IsTrusted reports whether a cwd (or an ancestor) is trusted.
func (s *Store) IsTrusted(cwd string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	cur, err := filepath.Abs(cwd)
	if err != nil {
		return false
	}
	for {
		if v, ok := s.trust[cur]; ok && v {
			return true
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			return false
		}
		cur = parent
	}
}

// SetTrust persists a trust decision for cwd.
func (s *Store) SetTrust(cwd string, trusted bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	abs, err := filepath.Abs(cwd)
	if err != nil {
		return err
	}
	if s.trust == nil {
		s.trust = trustFile{}
	}
	s.trust[abs] = trusted
	s.save()
	return nil
}

func configDir() string {
	if v := os.Getenv("XDG_CONFIG_HOME"); v != "" {
		return v
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".config")
	}
	return ""
}
