# pi-go

A Go port of [earendil-works/pi](https://github.com/earendil-works/pi), a TypeScript
AI agent toolkit. pi-go uses the official Go LLM SDKs
([anthropic-sdk-go](https://github.com/anthropics/anthropic-sdk-go) and
[openai-go](https://github.com/openai/openai-go)) as the HTTP layer beneath a
self-written pi-style unified abstraction.

## Status

Implemented (M1–M6):

- **ai** — unified abstraction: `Context`, `Message` (user/assistant/toolResult),
  content blocks (text/thinking/image/toolCall), `Usage`, `Model`, `Tool`, the
  streaming `AssistantMessageEvent` protocol, and an `EventStream` primitive.
- **provider** — `Anthropic` (Messages API), `OpenAI` (Chat Completions), and
  `OpenAIResponses` (Responses API). The OpenAI providers are OpenAI-compatible
  via `option.WithBaseURL`. All translate vendor stream events into the pi
  protocol and harden against gateways that omit stream-stop events.
- **agent** — the agent loop: `runLoop` (turns + follow-ups), tool execution
  (sequential / parallel), before/after-tool hooks, steering/follow-up queues,
  context transform, abort via `context.Context`.
- **harness** — stateful `AgentHarness` (Prompt/Steer/FollowUp/Abort/Subscribe +
  phase machine) with session persistence (in-memory and jsonl + meta sidecar),
  compaction-aware context rebuild, and union JSON (de)serialization.
- **tools** — read, bash, write, edit, grep, glob.
- **mcp** — minimal Model Context Protocol client over stdio; launches an MCP
  server as a child process and exposes its tools as agent tools (a pi-go
  addition; pi itself does not ship MCP).
- **tui** — differential-rendered terminal view: keeps the previous frame and
  only rewrites changed rows, avoiding full-screen flicker. A small,
  dependency-free analogue of pi-tui's renderer.
- **skills / prompt-templates** — load `SKILL.md` / `.md` files (YAML
  frontmatter) from disk; `harness.Skill(name)` injects a skill block,
  `harness.PromptFromTemplate(name, args)` expands positional `$1`/`$@`
  placeholders. Available skills are listed in the system prompt.
- **CLI** — single-prompt mode (`--prompt`), interactive REPL (multi-turn with
  persisted session), `--tui` differential view, `.env` auto-load,
  `.pi-go/config.json` defaults, and `--mcp` / `--skills-dir` /
  `--templates-dir` flags.

## Install

```bash
go build ./cmd/pi
```

Requires Go 1.24+.

## Configure

Secrets come from environment variables (or a gitignored `.env` file, auto-loaded):

```
# Anthropic
ANTHROPIC_API_KEY=...           # or ANTHROPIC_AUTH_TOKEN
ANTHROPIC_BASE_URL=...          # optional, for a compatible gateway

# OpenAI / OpenAI-compatible (OpenRouter, vLLM, Ollama, newapi, ...)
OPENAI_API_KEY=...              # or OPENAI_AUTH_TOKEN
OPENAI_BASE_URL=...             # optional
```

Optional `.pi-go/config.json` for defaults:

```json
{ "provider": "anthropic", "model": "claude-haiku-4-5" }
```

## Use

```bash
# Single prompt
./pi --prompt "Read go.mod and tell me the module name" --model claude-haiku-4-5

# Interactive multi-turn REPL (session persisted to .pi-go/sessions/)
./pi --session mywork

# OpenAI-compatible endpoint
./pi --provider openai --base-url http://localhost:11434/v1 --model llama3 --prompt "..."

# OpenAI Responses API
./pi --provider openai-responses --model gpt-4o --prompt "..."

# Differential-rendered TUI view
./pi --tui --prompt "..."

# With an MCP server providing extra tools
./pi --mcp "npx -y @modelcontextprotocol/server-filesystem ." --prompt "..."

# Load skills and prompt templates from custom dirs
./pi --skills-dir .skills --templates-dir .templates --session work
```

Flags: `--prompt`, `--provider`, `--model`, `--base-url`, `--system`,
`--session`, `--verbose`, `--tui`, `--mcp`, `--skills-dir`, `--templates-dir`.
In the REPL, `/help`, `/exit`, `/clear`.

## Layout

```
cmd/pi/           CLI entry, REPL, event rendering, .env + config loaders
internal/ai/      unified types, EventStream, validation, provider adapters
internal/agent/   agent loop + tool execution
internal/harness/ AgentHarness + session (memory + jsonl) + compaction + skills + templates
internal/mcp/     minimal MCP stdio client (exposes MCP tools as agent tools)
internal/tools/   read, bash, write, edit, grep, glob
internal/tui/     differential-rendered terminal view
```
