package harness

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadSkills(t *testing.T) {
	dir := t.TempDir()
	// SKILL.md with frontmatter.
	os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte(
		"---\nname: my-skill\ndescription: A test skill.\n---\nDo the thing.\n"), 0o644)
	// A nested skill.
	os.MkdirAll(filepath.Join(dir, "sub"), 0o755)
	os.WriteFile(filepath.Join(dir, "sub", "SKILL.md"), []byte(
		"---\nname: nested\ndisable-model-invocation: true\n---\nNested content.\n"), 0o644)

	skills, diags := LoadSkills(dir)
	if len(diags) != 0 {
		t.Fatalf("unexpected diagnostics: %v", diags)
	}
	if len(skills) != 2 {
		t.Fatalf("expected 2 skills, got %d", len(skills))
	}
	want := map[string]bool{"my-skill": false, "nested": false}
	for _, s := range skills {
		if _, ok := want[s.Name]; !ok {
			t.Fatalf("unexpected skill: %s", s.Name)
		}
		want[s.Name] = true
		if s.Name == "my-skill" && s.Description != "A test skill." {
			t.Fatalf("unexpected description: %s", s.Description)
		}
		if s.Name == "nested" && !s.DisableModelInvocation {
			t.Fatal("nested skill should have DisableModelInvocation")
		}
	}
}

func TestFormatSkillInvocation(t *testing.T) {
	s := Skill{Name: "x", Content: "body", FilePath: "/a/b/SKILL.md"}
	out := FormatSkillInvocation(s, "extra")
	if !contains(out, "<skill name=\"x\"") || !contains(out, "body") || !contains(out, "extra") {
		t.Fatalf("unexpected invocation: %s", out)
	}
}

func TestSubstituteArgs(t *testing.T) {
	cases := []struct {
		content string
		args    []string
		want    string
	}{
		{"hello $1", []string{"world"}, "hello world"},
		{"$1 and $2", []string{"a", "b"}, "a and b"},
		{"$1 $1", []string{"only"}, "only only"},
		{"all: $@", []string{"a", "b", "c"}, "all: a b c"},
		{"all: $ARGUMENTS", []string{"a", "b"}, "all: a b"},
		{"${@:2}", []string{"a", "b", "c"}, "b c"},
		{"${@:1:2}", []string{"a", "b", "c"}, "a b"},
		{"missing $5", []string{"a"}, "missing "},
	}
	for i, c := range cases {
		got := SubstituteArgs(c.content, c.args)
		if got != c.want {
			t.Errorf("case %d: SubstituteArgs(%q, %v) = %q, want %q", i, c.content, c.args, got, c.want)
		}
	}
}

func TestLoadPromptTemplates(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "review.md"), []byte(
		"---\nname: review\ndescription: Review code.\n---\nReview $1 please.\n"), 0o644)
	templates, diags := LoadPromptTemplates(dir)
	if len(diags) != 0 || len(templates) != 1 {
		t.Fatalf("expected 1 template, got %d (diags: %v)", len(templates), diags)
	}
	if templates[0].Name != "review" {
		t.Fatalf("unexpected name: %s", templates[0].Name)
	}
	out := FormatPromptTemplateInvocation(templates[0], []string{"main.go"})
	if out != "Review main.go please." {
		t.Fatalf("unexpected formatted template: %q", out)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
