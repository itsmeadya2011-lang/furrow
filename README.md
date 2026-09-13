# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow start "Add JWT auth" --max-cycles 5
furrow web
```

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.

## Features

- **Parallel task execution** — Planner breaks goals into 1-5 independent tasks executed concurrently
- **Auto-retry on LLM failures** — Exponential backoff retry on API errors (up to 3 attempts)
- **Structured logging** — JSON-formatted logs via structlog, configurable log level (`FURROW_LOG_LEVEL`)
- **Test-driven cycles** — Automatic test detection across pytest, npm, pnpm, yarn, cargo, go; failure feeds next cycle
- **Max cycles control** — `--max-cycles` CLI flag to limit loop iterations
- **Web UI** — WebSocket-based real-time monitoring via `furrow web`
