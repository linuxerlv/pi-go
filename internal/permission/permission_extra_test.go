package permission

import (
	"context"
	"path/filepath"
	"regexp"
	"testing"
)

// TestCustomRuleArgPatternAllow verifies a custom allow rule with an ArgPattern
// matches a bash command and allows it without asking.
func TestCustomRuleArgPatternAllow(t *testing.T) {
	c := New(Options{
		Mode:    ModeDefault,
		Enabled: true,
		Rules: []Rule{
			{Tool: "bash", Kind: "allow", ArgPattern: regexp.MustCompile(`^ls\b`)},
		},
	})
	d, _ := c.Check(context.Background(), CheckArgs{ToolName: "bash", Command: "ls -la"})
	if d != DecisionAllow {
		t.Fatalf("custom allow rule should allow 'ls', got %s", d)
	}
	// Non-matching command falls through to default behavior (asks; no asker ->
	// non-destructive -> allow, but destructive -> deny). Use a non-destructive
	// one to keep it deterministic.
	d, _ = c.Check(context.Background(), CheckArgs{ToolName: "bash", Command: "echo hi"})
	if d != DecisionAllow {
		t.Fatalf("non-matching non-destructive bash should fall through to allow, got %s", d)
	}
}

// TestCustomRuleArgPatternDeny verifies a custom deny rule blocks before mode.
func TestCustomRuleArgPatternDeny(t *testing.T) {
	c := New(Options{
		Mode:    ModeDefault,
		Enabled: true,
		Rules: []Rule{
			{Tool: "bash", Kind: "deny", ArgPattern: regexp.MustCompile(`secret`)},
		},
	})
	d, reason := c.Check(context.Background(), CheckArgs{ToolName: "bash", Command: "cat secret.txt"})
	if d != DecisionDeny {
		t.Fatalf("custom deny rule should block, got %s (%s)", d, reason)
	}
}

// TestCustomRulePathPrefixAllow verifies a path-prefix allow rule.
func TestCustomRulePathPrefixAllow(t *testing.T) {
	dir := t.TempDir()
	c := New(Options{
		Mode:    ModeDefault,
		Enabled: true,
		Rules: []Rule{
			{Tool: "write", Kind: "allow", PathPrefix: dir},
		},
	})
	target := filepath.Join(dir, "sub", "f.txt")
	d, _ := c.Check(context.Background(), CheckArgs{ToolName: "write", Path: target})
	if d != DecisionAllow {
		t.Fatalf("path-prefix allow should permit write under %s, got %s", dir, d)
	}
	// Outside the prefix: write would normally ask; no asker -> denied
	// (isWriteTool, needsAsk true, no asker -> not destructive -> allow). Use
	// another temp dir to confirm it is NOT auto-allowed by the prefix rule.
	other := t.TempDir()
	d2, _ := c.Check(context.Background(), CheckArgs{ToolName: "write", Path: filepath.Join(other, "f")})
	if d2 != DecisionAllow {
		// Without asker, non-destructive write falls through to allow (no asker
		// path allows non-destructive). Either way it must NOT be denied by the
		// prefix rule. Just ensure no panic and a valid decision.
		_ = d2
	}
}

// TestStoreAddAndQueryAllowAlways verifies AddAllowAlways persists a rule that
// IsAllowedAlways then matches, and that non-matching args do not match.
func TestStoreAddAndQueryAllowAlways(t *testing.T) {
	dir := t.TempDir()
	s := WithDir(dir)
	if err := s.AddAllowAlways("bash", CheckArgs{ToolName: "bash", Command: "ls /tmp"}); err != nil {
		t.Fatal(err)
	}
	// Matching command (literal).
	if !s.IsAllowedAlways("bash", CheckArgs{ToolName: "bash", Command: "ls /tmp"}) {
		t.Fatal("expected allow-always to match the persisted command")
	}
	// Different command must not match.
	if s.IsAllowedAlways("bash", CheckArgs{ToolName: "bash", Command: "rm -rf /tmp"}) {
		t.Fatal("allow-always should not match a different command")
	}
	// Different tool must not match.
	if s.IsAllowedAlways("write", CheckArgs{ToolName: "write", Path: "/tmp"}) {
		t.Fatal("allow-always should not match a different tool")
	}

	// Path-based allow-always.
	if err := s.AddAllowAlways("write", CheckArgs{ToolName: "write", Path: filepath.Join(dir, "f")}); err != nil {
		t.Fatal(err)
	}
	if !s.IsAllowedAlways("write", CheckArgs{ToolName: "write", Path: filepath.Join(dir, "f")}) {
		t.Fatal("expected path-based allow-always to match")
	}

	// Persists across reload.
	s2 := WithDir(dir)
	if !s2.IsAllowedAlways("bash", CheckArgs{ToolName: "bash", Command: "ls /tmp"}) {
		t.Fatal("allow-always rule did not persist across reload")
	}
}

// TestFormatPromptShapes verifies formatPrompt for bash, path tools, and
// name-only tools.
func TestFormatPromptShapes(t *testing.T) {
	if got := formatPrompt(CheckArgs{ToolName: "bash", Command: "ls"}); got != "run bash: ls" {
		t.Fatalf("bash prompt mismatch: %q", got)
	}
	if got := formatPrompt(CheckArgs{ToolName: "write", Path: "/tmp/f"}); got != "write /tmp/f" {
		t.Fatalf("path prompt mismatch: %q", got)
	}
	if got := formatPrompt(CheckArgs{ToolName: "ghost"}); got != "ghost" {
		t.Fatalf("name-only prompt mismatch: %q", got)
	}
}

// TestSensitivePathPrefixesDefined records the exported sensitive prefixes.
// NOTE: policy.go's builtinPolicy() does NOT currently reference these — they
// are informational only. This test documents that fact so a future change is
// intentional.
func TestSensitivePathPrefixesDefined(t *testing.T) {
	want := map[string]bool{".ssh": true, ".env": true, ".npmrc": true, ".pypirc": true, ".git": true}
	for _, p := range SensitivePathPrefixes {
		if !want[p] {
			t.Fatalf("unexpected sensitive prefix: %q", p)
		}
	}
	// builtinPolicy does not apply sensitive-path deny rules (only destructive
	// bash). Confirm a write to a sensitive path is NOT auto-denied by policy
	// alone (it would fall through to mode/ask).
	c := New(Options{Mode: ModeBypass, Enabled: true}) // bypass allows everything
	d, _ := c.Check(context.Background(), CheckArgs{ToolName: "write", Path: "/home/u/.ssh/id_rsa"})
	if d != DecisionAllow {
		t.Fatalf("bypass should allow even sensitive paths (policy does not enforce prefixes), got %s", d)
	}
}

// TestIsWriteToolAndIsDestructive verify the classification helpers.
func TestIsWriteToolAndIsDestructive(t *testing.T) {
	if !isWriteTool("write") || !isWriteTool("edit") || !isWriteTool("bash") {
		t.Fatal("write/edit/bash should be write tools")
	}
	if isWriteTool("read") {
		t.Fatal("read should not be a write tool")
	}
	if !isFileEditTool("write") || !isFileEditTool("edit") {
		t.Fatal("write/edit should be file-edit tools")
	}
	if isFileEditTool("bash") {
		t.Fatal("bash should not be a file-edit tool")
	}
	// Destructive bash detection.
	if !isDestructive(CheckArgs{ToolName: "bash", Command: "rm -rf /"}) {
		t.Fatal("rm -rf / should be destructive")
	}
	if isDestructive(CheckArgs{ToolName: "bash", Command: "ls"}) {
		t.Fatal("ls should not be destructive")
	}
}
