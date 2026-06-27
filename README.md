# pi-go

A Go port of [earendil-works/pi](https://github.com/earendil-works/pi), a TypeScript
AI agent toolkit. pi-go uses the official Go LLM SDKs
([anthropic-sdk-go](https://github.com/anthropics/anthropic-sdk-go) and
[openai-go](https://github.com/openai/openai-go)) as the HTTP layer beneath a
self-written pi-style unified abstraction.

## Status

Implemented (M1–M5):

- **ai** — unified abstraction: `Context`, `Message` (user/assistant/toolResult),
  content blocks (text/thinking/image/toolCall), `Usage`, `Model`, `Tool`, the
  streaming `AssistantMessageEvent` protocol, and an `EventStream` primitive.
- **provider** — `Anthropic` (Messages API) and `OpenAI` (Chat Completions,
  OpenAI-compatible via `option.WithBaseURL`). Both translate vendor stream
  events into the pi protocol and harden against gateways that omit stream-stop
  events.
- **agent** — the agent loop: `runLoop` (turns + follow-ups), tool execution
  (sequential / parallel), before/after-tool hooks, steering/follow-up queues,
  context transform, abort via `context.Context`.
- **harness** — stateful `AgentHarness` (Prompt/Steer/FollowUp/Abort/Subscribe +
  phase machine) with session persistence (in-memory and jsonl + meta sidecar),
  compaction-aware context rebuild, and union JSON (de)serialization.
- **tools** — read, bash, write, edit, grep, glob.
- **CLI** — single-prompt mode (`--prompt`), interactive REPL (multi-turn with
  persisted session), `.env` auto-load, and `.pi-go/config.json` defaults.

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
```

Flags: `--prompt`, `--provider`, `--model`, `--base-url`, `--system`,
`--session`, `--verbose`. In the REPL, `/help`, `/exit`, `/clear`.

## Layout

```
cmd/pi/           CLI entry, REPL, event rendering, .env + config loaders
internal/ai/      unified types, EventStream, validation, provider adapters
internal/agent/   agent loop + tool execution
internal/harness/ AgentHarness + session (memory + jsonl) + compaction
internal/tools/   read, bash, write, edit, grep, glob
```
