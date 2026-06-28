package permission

import (
	"context"
	"path/filepath"
	"testing"
)

func TestModeBypassAllowsEverything(t *testing.T) {
	c := New(Options{Mode: ModeBypass, Enabled: true})
	d, _ := c.Check(context.Background(), CheckArgs{ToolName: "bash", Command: "rm -rf /tmp/x"})
	if d != DecisionAllow {
		t.Fatalf("bypass should allow, got %s", d)
	}
}

func TestDestructiveBashDenied(t *testing.T) {
	// Even with no asker, hard-destructive commands are denied by policy.
	c := New(Options{Mode: ModeDefault, Enabled: true})
	cases := []string{
		"rm -rf /",
		"git push --force origin main",
		"git push -f",
		"dd if=/dev/zero of=/dev/sda",
		"chmod -R 777 /",
	}
	for _, cmd := range cases {
		d, reason := c.Check(context.Background(), CheckArgs{ToolName: "bash", Command: cmd})
		if d != DecisionDeny {
			t.Fatalf("expected deny for %q, got %s (%s)", cmd, d, reason)
		}
	}
}

func TestModePlanBlocksWrites(t *testing.T) {
	c := New(Options{Mode: ModePlan, Enabled: true})
	d, _ := c.Check(context.Background(), CheckArgs{ToolName: "write", Path: "/tmp/f"})
	if d != DecisionDeny {
		t.Fatalf("plan should deny write, got %s", d)
	}
	// Read is allowed in plan mode.
	d, _ = c.Check(context.Background(), CheckArgs{ToolName: "read", Path: "/tmp/f"})
	if d != DecisionAllow {
		t.Fatalf("plan should allow read, got %s", d)
	}
}

func TestModeAcceptEditsAllowsFileEdits(t *testing.T) {
	c := New(Options{Mode: ModeAcceptEdits, Enabled: true})
	d, _ := c.Check(context.Background(), CheckArgs{ToolName: "edit", Path: "/tmp/f"})
	if d != DecisionAllow {
		t.Fatalf("acceptEdits should allow edit, got %s", d)
	}
	// bash still asks (no asker -> deny destructive, but non-destructive would ask).
	// A non-destructive bash with no asker: needsAsk true, no asker -> allow.
	d, _ = c.Check(context.Background(), CheckArgs{ToolName: "bash", Command: "ls"})
	if d != DecisionAllow {
		t.Fatalf("acceptEdits non-destructive bash with no asker should allow, got %s", d)
	}
}

func TestAskerAllow(t *testing.T) {
	asked := false
	c := New(Options{
		Mode:    ModeDefault,
		Enabled: true,
		Asker: func(ctx context.Context, prompt string) (string, error) {
			asked = true
			return "allow", nil
		},
	})
	d, _ := c.Check(context.Background(), CheckArgs{ToolName: "bash", Command: "ls -la"})
	if d != DecisionAllow {
		t.Fatalf("expected allow after asker says allow, got %s", d)
	}
	if !asked {
		t.Fatal("asker should have been called for bash")
	}
}

func TestAskerDeny(t *testing.T) {
	c := New(Options{
		Mode:    ModeDefault,
		Enabled: true,
		Asker: func(ctx context.Context, prompt string) (string, error) {
			return "deny", nil
		},
	})
	d, reason := c.Check(context.Background(), CheckArgs{ToolName: "bash", Command: "ls"})
	if d != DecisionDeny {
		t.Fatalf("expected deny, got %s", d)
	}
	if reason == "" {
		t.Fatal("expected a reason on deny")
	}
}

func TestAllowAlwaysPersists(t *testing.T) {
	dir := t.TempDir()
	store := WithDir(dir)
	c := New(Options{Mode: ModeDefault, Enabled: true, Store: store, Asker: func(ctx context.Context, p string) (string, error) {
		return "allow-always", nil
	}})

	// First check asks and persists allow-always.
	d, _ := c.Check(context.Background(), CheckArgs{ToolName: "bash", Command: "ls /tmp"})
	if d != DecisionAllow {
		t.Fatalf("expected allow, got %s", d)
	}

	// Reload store from disk; the rule must persist.
	store2 := WithDir(dir)
	c2 := New(Options{Mode: ModeDefault, Enabled: true, Store: store2})
	// No asker: if allow-always didn't persist, bash would fall to allow (no
	// asker, non-destructive) — so use a destructive-ish check that would ask.
	// Better: verify IsAllowedAlways directly.
	if !store2.IsAllowedAlways("bash", CheckArgs{ToolName: "bash", Command: "ls /tmp"}) {
		t.Fatal("allow-always rule did not persist")
	}
	_ = c2
}

func TestTrustStore(t *testing.T) {
	dir := t.TempDir()
	store := WithDir(dir)
	cwd := filepath.Join(dir, "project")
	if store.IsTrusted(cwd) {
		t.Fatal("should not be trusted initially")
	}
	if err := store.SetTrust(cwd, true); err != nil {
		t.Fatal(err)
	}
	if !store.IsTrusted(cwd) {
		t.Fatal("should be trusted after SetTrust")
	}
	// Subdirectory inherits trust via ancestor walk.
	sub := filepath.Join(cwd, "sub", "dir")
	if !store.IsTrusted(sub) {
		t.Fatal("subdir should inherit ancestor trust")
	}
	// Reload persists.
	store2 := WithDir(dir)
	if !store2.IsTrusted(cwd) {
		t.Fatal("trust did not persist")
	}
}

func TestDisabledCheckerAllowsAll(t *testing.T) {
	c := New(Options{Mode: ModeDefault, Enabled: false})
	d, _ := c.Check(context.Background(), CheckArgs{ToolName: "bash", Command: "rm -rf /"})
	if d != DecisionAllow {
		t.Fatalf("disabled checker should allow all, got %s", d)
	}
}
