package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadDotEnvParsesAndRespectsExisting(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".env")
	content := "# a comment\n\nKEY_NEW=value1\nKEY_EXISTING=fromfile\nQUOTED=\"hello world\"\nSINGLE='sp'\nBADLINE_NOEQ\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	t.Setenv("KEY_EXISTING", "fromenv")
	os.Unsetenv("KEY_NEW")
	t.Cleanup(func() { os.Unsetenv("KEY_NEW") })
	os.Unsetenv("QUOTED")
	t.Cleanup(func() { os.Unsetenv("QUOTED") })
	os.Unsetenv("SINGLE")
	t.Cleanup(func() { os.Unsetenv("SINGLE") })

	loadDotEnv(path)

	if got := os.Getenv("KEY_NEW"); got != "value1" {
		t.Fatalf("KEY_NEW should be set from file, got %q", got)
	}
	if got := os.Getenv("KEY_EXISTING"); got != "fromenv" {
		t.Fatalf("KEY_EXISTING should keep env value, got %q", got)
	}
	if got := os.Getenv("QUOTED"); got != "hello world" {
		t.Fatalf("QUOTED should be unquoted, got %q", got)
	}
	if got := os.Getenv("SINGLE"); got != "sp" {
		t.Fatalf("SINGLE should be unquoted, got %q", got)
	}
}

func TestLoadDotEnvMissingFileIsNoOp(t *testing.T) {
	loadDotEnv(filepath.Join(t.TempDir(), "does-not-exist"))
}
