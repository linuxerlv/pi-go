package harness

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Skill is a markdown skill file loaded from disk. Skills inject curated
// instructions into a prompt when explicitly invoked.
type Skill struct {
	Name        string
	Description string
	Content     string
	FilePath    string
	// DisableModelInvocation hides the skill from model-visible skill lists while
	// still allowing explicit Skill(name) invocation.
	DisableModelInvocation bool
}

// LoadSkills walks the given directories recursively and loads SKILL.md files
// (and direct root .md files) as skills. Each skill file is markdown with an
// optional YAML frontmatter block (--- delimited) carrying name/description.
// Missing directories are skipped. Returns loaded skills and parse diagnostics.
func LoadSkills(dirs ...string) ([]Skill, []Diagnostic) {
	var skills []Skill
	var diags []Diagnostic
	for _, dir := range dirs {
		info, err := os.Stat(dir)
		if err != nil {
			continue
		}
		if !info.IsDir() {
			continue
		}
		err = filepath.WalkDir(dir, func(p string, d os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			if d.IsDir() {
				base := d.Name()
				if base == ".git" || base == "node_modules" || base == ".pi-go" {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(d.Name(), ".md") {
				return nil
			}
			// SKILL.md anywhere, or any .md directly at the root of a skill dir.
			if d.Name() == "SKILL.md" || filepath.Dir(p) == dir {
				s, d := loadSkillFile(p)
				if d != nil {
					diags = append(diags, *d)
				}
				if s != nil {
					skills = append(skills, *s)
				}
			}
			return nil
		})
		if err != nil {
			diags = append(diags, Diagnostic{Code: "list_failed", Message: err.Error(), Path: dir})
		}
	}
	return skills, diags
}

// loadSkillFile reads a markdown file, parsing YAML frontmatter for
// name/description, and returns the skill.
func loadSkillFile(path string) (*Skill, *Diagnostic) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, &Diagnostic{Code: "read_failed", Message: err.Error(), Path: path}
	}
	name, desc, disableInvoke, content, perr := parseFrontmatter(string(b))
	if perr != nil {
		return nil, &Diagnostic{Code: "parse_failed", Message: perr.Error(), Path: path}
	}
	if name == "" {
		// Derive name from filename.
		base := strings.TrimSuffix(filepath.Base(path), ".md")
		if base == "SKILL" {
			name = filepath.Base(filepath.Dir(path))
		} else {
			name = base
		}
	}
	return &Skill{
		Name:                   name,
		Description:            desc,
		Content:                content,
		FilePath:               path,
		DisableModelInvocation: disableInvoke,
	}, nil
}

// Diagnostic is a non-fatal warning produced while loading skills/templates.
type Diagnostic struct {
	Type    string // "warning"
	Code    string
	Message string
	Path    string
}

// frontmatterRe splits a markdown file into (yaml frontmatter, body).
var frontmatterRe = regexp.MustCompile(`(?s)^\s*---\s*\n(.*?)\n---\s*\n?(.*)$`)

// parseFrontmatter extracts simple YAML keys (name, description,
// disable-model-invocation) from a leading frontmatter block.
func parseFrontmatter(text string) (name, desc string, disableInvoke bool, body string, err error) {
	m := frontmatterRe.FindStringSubmatch(text)
	if m == nil {
		return "", "", false, text, nil
	}
	yamlBlock := m[1]
	body = strings.TrimSpace(m[2])
	for _, line := range strings.Split(yamlBlock, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		value = strings.Trim(value, `"'`)
		switch key {
		case "name":
			name = value
		case "description":
			desc = value
		case "disable-model-invocation":
			disableInvoke = value == "true"
		}
	}
	return name, desc, disableInvoke, body, nil
}

// FormatSkillInvocation builds the prompt block that injects a skill into the
// conversation. Mirrors pi-agent-core's formatSkillInvocation.
func FormatSkillInvocation(skill Skill, additionalInstructions string) string {
	dir := filepath.Dir(skill.FilePath)
	block := fmt.Sprintf("<skill name=%q location=%q>\nReferences are relative to %s.\n\n%s\n</skill>",
		skill.Name, skill.FilePath, dir, skill.Content)
	if additionalInstructions != "" {
		return block + "\n\n" + additionalInstructions
	}
	return block
}
