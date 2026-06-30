package main

import (
	"strings"
	"testing"

	"github.com/linuxerlv/pi-go/internal/ai"
	"github.com/linuxerlv/pi-go/internal/ai/provider"
)

func TestSplitShellArgs(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"npx -y @modelcontextprotocol/server", []string{"npx", "-y", "@modelcontextprotocol/server"}},
		{`echo "hello world"`, []string{"echo", "hello world"}},
		{"  trim   me  ", []string{"trim", "me"}},
		{`a "b c" d`, []string{"a", "b c", "d"}},
		{"", nil},
	}
	for _, c := range cases {
		got, err := splitShellArgs(c.in)
		if err != nil {
			t.Fatalf("splitShellArgs(%q) error: %v", c.in, err)
		}
		if len(got) != len(c.want) {
			t.Fatalf("splitShellArgs(%q) = %v, want %v", c.in, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("splitShellArgs(%q)[%d] = %q, want %q", c.in, i, got[i], c.want[i])
			}
		}
	}
}

func TestSplitShellArgsUnterminatedQuote(t *testing.T) {
	_, err := splitShellArgs(`echo "unclosed`)
	if err == nil || !strings.Contains(err.Error(), "unterminated quote") {
		t.Fatalf("expected unterminated-quote error, got: %v", err)
	}
}

func TestStringsReplaceAll(t *testing.T) {
	if got := stringsReplaceAll("a.b.c", ".", "/"); got != "a/b/c" {
		t.Fatalf("unexpected: %q", got)
	}
}

// dim/red strip ANSI when stdout is not a TTY (the case under `go test`).
func TestAnsiHelpersStripWhenNotTTY(t *testing.T) {
	if isTTY() {
		t.Skip("stdout is a TTY in this environment; skipping non-TTY assertions")
	}
	if got := dim("x"); got != "x" {
		t.Fatalf("dim should be stripped when not a TTY, got %q", got)
	}
	if got := red("x"); got != "x" {
		t.Fatalf("red should be stripped when not a TTY, got %q", got)
	}
}

// --- provider resolution ---

func TestResolveProviderUnknown(t *testing.T) {
	_, _, err := resolveProvider("bogus", "")
	if err == nil || !strings.Contains(err.Error(), "unknown provider") {
		t.Fatalf("expected unknown-provider error, got: %v", err)
	}
}

func TestResolveProviderNoEnv(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENAI_AUTH_TOKEN", "")
	_, _, err := resolveProvider("", "")
	if err == nil || !strings.Contains(err.Error(), "no provider configured") {
		t.Fatalf("expected no-provider-configured error, got: %v", err)
	}
}

func TestResolveProviderAnthropicAutoDetect(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "sk-test")
	t.Setenv("OPENAI_API_KEY", "")
	prov, model, err := resolveProvider("", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model != "claude-haiku-4-5" {
		t.Fatalf("expected default anthropic model, got %q", model)
	}
	if prov == nil {
		t.Fatal("expected non-nil provider")
	}
}

func TestResolveProviderOpenAIExplicit(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-test")
	prov, model, err := resolveProvider("openai", "http://localhost:11434/v1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model != "gpt-4o-mini" {
		t.Fatalf("expected default openai model, got %q", model)
	}
	if prov == nil {
		t.Fatal("expected non-nil provider")
	}
}

func TestResolveProviderOpenAIMissingKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENAI_AUTH_TOKEN", "")
	_, _, err := resolveProvider("openai", "")
	if err == nil || !strings.Contains(err.Error(), "is not set") {
		t.Fatalf("expected missing-key error, got: %v", err)
	}
}

func TestOpenAIKeyFromEnv(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "primary")
	t.Setenv("OPENAI_AUTH_TOKEN", "fallback")
	if k := openAIKeyFromEnv(); k != "primary" {
		t.Fatalf("expected primary key, got %q", k)
	}
	t.Setenv("OPENAI_API_KEY", "")
	if k := openAIKeyFromEnv(); k != "fallback" {
		t.Fatalf("expected fallback token, got %q", k)
	}
}

func TestRegisterDynamicModelAnthropic(t *testing.T) {
	prov := provider.NewAnthropic("sk-test")
	m := registerDynamicModel(prov, "custom-claude")
	if m.ID != "custom-claude" || m.API != ai.APIAnthropicMessages {
		t.Fatalf("unexpected model: %+v", m)
	}
	if got, ok := prov.GetModel("custom-claude"); !ok || got.ID != "custom-claude" {
		t.Fatalf("model not registered: %+v", got)
	}
}

func TestRegisterDynamicModelUnknownProvider(t *testing.T) {
	m := registerDynamicModel(nil, "x")
	if m.ID != "x" || m.Name != "x" {
		t.Fatalf("unexpected fallback model: %+v", m)
	}
}

// --- mode routing ---

func TestResolveModePrecedence(t *testing.T) {
	cases := []struct {
		name string
		d    deps
		want string
	}{
		{"tui wins all", deps{useTUI: true, prompt: "x", orchestrate: true, sessionID: "s"}, "tui"},
		{"repl when no prompt", deps{prompt: ""}, "repl"},
		{"orchestrate when prompt+flag", deps{prompt: "x", orchestrate: true}, "orchestrate"},
		{"harness-single when session", deps{prompt: "x", sessionID: "s"}, "harness"},
		{"bare loop otherwise", deps{prompt: "x"}, "bare"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := resolveMode(c.d)
			if got := runnerName(r); got != c.want {
				t.Fatalf("resolveMode = %s, want %s", got, c.want)
			}
		})
	}
}

func runnerName(r modeRunner) string {
	switch r.(type) {
	case tuiRunner:
		return "tui"
	case replRunner:
		return "repl"
	case orchestrateRunner:
		return "orchestrate"
	case harnessSingleRunner:
		return "harness"
	case bareLoopRunner:
		return "bare"
	}
	return "?"
}

func TestResolveSessionID(t *testing.T) {
	if got := resolveSessionID(deps{sessionID: "abc"}); got != "abc" {
		t.Fatalf("expected explicit id, got %q", got)
	}
	if got := resolveSessionID(deps{}); got != "default" {
		t.Fatalf("expected default, got %q", got)
	}
}
