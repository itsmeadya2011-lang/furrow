# Furrow

Autonomous coding agent with an infinite parallel development loop.

Furrow breaks a high-level goal into parallelizable tasks, executes them with
specialized worker agents, verifies the result with a tester agent, and repeats
until the goal is complete or the user stops it.

## Features

- **Infinite loop** — plans, executes, tests, and repeats until the goal is done.
- **Parallel execution** — up to `FURROW_MAX_PARALLEL_TASKS` tasks run concurrently each cycle.
- **Role-based agents** — a planner decomposes goals, workers implement single
  tasks, and a tester validates the outcome.
- **Multiple providers** — Anthropic, OpenAI, or Ollama backends.
- **Two interfaces** — a CLI (`furrow start`) and a web UI (`furrow web`) with
  live WebSocket output.

## Installation

```bash
pip install furrow
```

For development:

```bash
pip install -e ".[dev]"
```

## Configuration

Furrow reads configuration from environment variables (prefix `FURROW_`) or a
`.env` file in the current directory. See [`.env.example`](.env.example) for the
full list of options.

| Variable | Default | Description |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, or `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model for all agent roles |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the planner agent |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the worker agent |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the tester agent |
| `FURROW_ANTHROPIC_API_KEY` | — | Anthropic API key |
| `FURROW_OPENAI_API_KEY` | — | OpenAI API key |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama base URL |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Max tasks executed in parallel per cycle |
| `FURROW_MAX_CYCLES` | `0` | Max development cycles (`0` = unlimited) |
| `FURROW_LOG_LEVEL` | `INFO` | Logging level |

## Usage

### CLI

```bash
# Start the autonomous loop with a goal
furrow start "Add JWT auth to the API"

# Or pass a model override
furrow start "Add JWT auth to the API" --model claude-sonnet-4-20250514

# Start the web UI
furrow web
```

### Web

Running `furrow web` serves a browser-based interface on `http://0.0.0.0:8000`.
Enter a goal in the input field and watch live output via WebSocket.

## Architecture

Furrow is organized into four agent roles orchestrated by an infinite loop:

1. **Planner** — decomposes a goal into independent, parallelizable tasks.
2. **Worker** — executes a single task with minimal, targeted changes.
3. **Tester** — runs tests, lint, and type checks; reports pass/fail.
4. **Orchestrator** — drives the loop: plan → execute → test → repeat.

See [`.kilo/FURROW.md`](.kilo/FURROW.md) for the full architecture overview.

## Testing

```bash
pytest
```

## License

MIT