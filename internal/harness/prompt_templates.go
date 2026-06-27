package harness

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// PromptTemplate is a markdown template loaded from disk. It can be formatted
// with positional arguments and turned into a prompt.
type PromptTemplate struct {
	Name        string
	Description string
	Content     string
}

// LoadPromptTemplates loads .md files from the given paths. Directory inputs
// load direct .md children (non-recursive); file inputs load explicitly.
// Missing paths and non-markdown files are skipped.
func LoadPromptTemplates(paths ...string) ([]PromptTemplate, []Diagnostic) {
	var templates []PromptTemplate
	var diags []Diagnostic
	for _, p := range paths {
		info, err := os.Stat(p)
		if err != nil {
			continue
		}
		if info.IsDir() {
			entries, err := os.ReadDir(p)
			if err != nil {
				diags = append(diags, Diagnostic{Code: "list_failed", Message: err.Error(), Path: p})
				continue
			}
			for _, e := range entries {
				if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
					continue
				}
				if t, d := loadTemplateFile(filepath.Join(p, e.Name())); t != nil {
					templates = append(templates, *t)
				} else if d != nil {
					diags = append(diags, *d)
				}
			}
		} else if strings.HasSuffix(p, ".md") {
			if t, d := loadTemplateFile(p); t != nil {
				templates = append(templates, *t)
			} else if d != nil {
				diags = append(diags, *d)
			}
		}
	}
	return templates, diags
}

func loadTemplateFile(path string) (*PromptTemplate, *Diagnostic) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, &Diagnostic{Code: "read_failed", Message: err.Error(), Path: path}
	}
	name, desc, _, content, perr := parseFrontmatter(string(b))
	if perr != nil {
		return nil, &Diagnostic{Code: "parse_failed", Message: perr.Error(), Path: path}
	}
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(path), ".md")
	}
	return &PromptTemplate{Name: name, Description: desc, Content: content}, nil
}

// argPosRe matches $1, $2, ... positional placeholders.
var argPosRe = regexp.MustCompile(`\$(\d+)`)

// argSliceRe matches ${@:start} or ${@:start:length}.
var argSliceRe = regexp.MustCompile(`\$\{@:(\d+)(?::(\d+))?\}`)

// SubstituteArgs expands template argument placeholders:
//
//	$1, $2, ...  -> the Nth positional arg (1-indexed)
//	${@:start}   -> args from start joined with spaces
//	${@:start:n} -> n args from start joined with spaces
//	$ARGUMENTS, $@ -> all args joined with spaces
//
// Mirrors pi-agent-core's substituteArgs.
func SubstituteArgs(content string, args []string) string {
	out := argSliceRe.ReplaceAllStringFunc(content, func(m string) string {
		sub := argSliceRe.FindStringSubmatch(m)
		start, _ := atoi(sub[1])
		start--
		if start < 0 {
			start = 0
		}
		if sub[2] != "" {
			length, _ := atoi(sub[2])
			end := start + length
			if end > len(args) {
				end = len(args)
			}
			if start > len(args) {
				start = len(args)
			}
			return strings.Join(args[start:end], " ")
		}
		if start > len(args) {
			start = len(args)
		}
		return strings.Join(args[start:], " ")
	})
	out = argPosRe.ReplaceAllStringFunc(out, func(m string) string {
		sub := argPosRe.FindStringSubmatch(m)
		n, _ := atoi(sub[1])
		if n >= 1 && n <= len(args) {
			return args[n-1]
		}
		return ""
	})
	allArgs := strings.Join(args, " ")
	out = strings.ReplaceAll(out, "$ARGUMENTS", allArgs)
	out = strings.ReplaceAll(out, "$@", allArgs)
	return out
}

// FormatPromptTemplateInvocation formats a template with positional arguments.
func FormatPromptTemplateInvocation(template PromptTemplate, args []string) string {
	return SubstituteArgs(template.Content, args)
}

func atoi(s string) (int, error) {
	n := 0
	for _, r := range s {
		if r < '0' || r > '9' {
			return 0, errInvalidInt
		}
		n = n*10 + int(r-'0')
	}
	return n, nil
}

var errInvalidInt = &parseErr{"invalid integer"}

type parseErr struct{ msg string }

func (e *parseErr) Error() string { return e.msg }
