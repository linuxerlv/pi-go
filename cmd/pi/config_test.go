package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfigMissingFile(t *testing.T) {
	c := loadConfig(filepath.Join(t.TempDir(), "nope.json"))
	if c.Provider != "" || c.Model != "" {
		t.Fatalf("expected zero config, got %+v", c)
	}
}

func TestLoadConfigValid(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	body := `{"provider":"openai","model":"gpt-4o","baseURL":"http://x","systemPrompt":"p"}`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	c := loadConfig(path)
	if c.Provider != "openai" || c.Model != "gpt-4o" || c.BaseURL != "http://x" || c.SystemPrompt != "p" {
		t.Fatalf("unexpected config: %+v", c)
	}
}

func TestLoadConfigInvalidJSONReturnsZero(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	c := loadConfig(path)
	if c.Provider != "" {
		t.Fatalf("expected zero config on invalid json, got %+v", c)
	}
}

func TestMergeConfigsOverrideWins(t *testing.T) {
	base := Config{Provider: "anthropic", Model: "old", BaseURL: "http://b", SystemPrompt: "base"}
	over := Config{Model: "new"}
	got := mergeConfigs(base, over)
	if got.Provider != "anthropic" || got.Model != "new" || got.BaseURL != "http://b" || got.SystemPrompt != "base" {
		t.Fatalf("merge mismatch: %+v", got)
	}
	got2 := mergeConfigs(base, Config{})
	if got2 != base {
		t.Fatalf("empty override should be no-op: %+v", got2)
	}
}

// TestLoadMergedConfigProjectOverridesUser writes both user and project config
// files and asserts project wins (priority: project > user).
func TestLoadMergedConfigProjectOverridesUser(t *testing.T) {
	// Redirect os.UserConfigDir via the env vars it reads per-platform.
	tmpHome := t.TempDir()
	t.Setenv("APPDATA", tmpHome)
	t.Setenv("XDG_CONFIG_HOME", tmpHome)
	t.Setenv("HOME", tmpHome)

	userPath := userConfigPath()
	if userPath == "" {
		t.Skip("userConfigPath empty in this environment; skipping")
	}
	if err := os.MkdirAll(filepath.Dir(userPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(userPath, []byte(`{"provider":"openai","model":"user-model"}`), 0o600); err != nil {
		t.Fatal(err)
	}

	projDir := filepath.Join(".pi-go")
	projPath := filepath.Join(projDir, "config.json")
	if err := os.MkdirAll(projDir, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(projDir) })
	if err := os.WriteFile(projPath, []byte(`{"model":"proj-model"}`), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg := loadMergedConfig()
	if cfg.Provider != "openai" {
		t.Fatalf("provider should come from user config, got %q", cfg.Provider)
	}
	if cfg.Model != "proj-model" {
		t.Fatalf("model should be overridden by project config, got %q", cfg.Model)
	}
}

func TestDefaultConfigPath(t *testing.T) {
	if p := defaultConfigPath(); p != filepath.Join(".pi-go", "config.json") {
		t.Fatalf("unexpected default config path: %q", p)
	}
}
