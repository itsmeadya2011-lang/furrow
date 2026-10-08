# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow web
```

## Configuration

Set one of `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.

All options can be set via `FURROW_`-prefixed environment variables or a `.env` file:

| Variable | Default | Description |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, or `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used for planning |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by workers |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the tester |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server (when provider=ollama) |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Max workers running concurrently |
| `FURROW_MAX_CYCLES` | `0` | Halt after N cycles (0 = unlimited) |
| `FURROW_REQUEST_TIMEOUT` | `60` | Per-request LLM timeout in seconds |

Transient LLM errors (network/rate-limit) are retried automatically with exponential backoff.