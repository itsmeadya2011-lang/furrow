# Furrow

Furrow is an autonomous coding agent that plans, implements, and tests changes to a codebase through an iterative loop. It coordinates a planner, a worker (which edits files using real tool calls), and a tester to move toward a goal while persisting state and optionally streaming progress via a web UI.

## Features

- Plans tasks with dependency tracking and parallel execution
- Worker actually implements code using file and command tools
- Project-aware test detection (pytest, npm/pnpm/yarn, cargo, go)
- Event-driven progress with a console UI and FastAPI web interface
- Atomic state persistence with resume support
- Offline test suite with no API keys required

## Install

Install in editable mode with dev dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### `furrow start "<goal>"`
Start (or resume) a run for the specified goal.

```bash
furrow start "Add JWT auth to the API"
```

Options:
- `--model` — override the LLM model
- `--max-cycles N` — maximum number of cycles (0 = unlimited)
- `--workspace PATH` — project directory to work in
- `--resume` — resume the persisted run instead of starting fresh

If `goal` is omitted, Furrow will prompt for it.

### `furrow status`
Show the persisted state of the last run for a workspace.

```bash
furrow status
furrow status --workspace /path/to/project
```

### `furrow resume`
Continue the persisted run.

```bash
furrow resume
furrow resume --workspace /path/to/project --max-cycles 5
```

### `furrow web`
Serve the web UI and stream orchestrator events.

```bash
furrow web
furrow web --host 127.0.0.1 --port 8000
```

The default web UI is available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

The web API refuses to start a second run while one is already in progress — both through `POST /api/start` and through the websocket.

## Configuration

Furrow reads environment variables with the `FURROW_` prefix (and a `.env` file if present). Available settings:

| Variable | Default | Description |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, or `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model for completions/chat |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model for the planner |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model for the worker |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model for the tester |
| `FURROW_PLANNER_PROVIDER` | (default) | Override provider for the planner role |
| `FURROW_WORKER_PROVIDER` | (default) | Override provider for the worker role |
| `FURROW_TESTER_PROVIDER` | (default) | Override provider for the tester role |
| `FURROW_PLANNER_API_KEY` | (default) | Override API key for the planner role |
| `FURROW_WORKER_API_KEY` | (default) | Override API key for the worker role |
| `FURROW_TESTER_API_KEY` | (default) | Override API key for the tester role |
| `FURROW_FALLBACK_CHAIN` | (none) | Ordered list of `provider:model` pairs to try when the primary call fails with a retryable error |
| `FURROW_ANTHROPIC_API_KEY` | (none) | Anthropic API key |
| `FURROW_OPENAI_API_KEY` | (none) | OpenAI API key |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama OpenAI-compatible base URL |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Maximum concurrent worker tasks per wave |
| `FURROW_MAX_CYCLES` | `0` | Maximum cycles to run (0 = unlimited) |
| `FURROW_MAX_WORKER_ITERATIONS` | `12` | Maximum tool-call iterations per worker task |
| `FURROW_MAX_RETRIES` | `3` | Retry attempts for transient LLM errors |
| `FURROW_MAX_TASK_RETRIES` | `1` | Times a failed task is retried before it is marked failed |
| `FURROW_LOG_LEVEL` | `INFO` | Logging level used by the CLI (honoured at startup) |
| `FURROW_REQUEST_TIMEOUT` | `120.0` | Request timeout in seconds |
| `FURROW_MAX_TOKENS` | `4096` | Max tokens for LLM completions |
| `FURROW_STATE_FILE` | `.furrow/state.json` | State file path (relative to workspace) |
| `FURROW_WORKSPACE` | current directory | Working directory for the agent |
| `FURROW_TRACK_COST` | `true` | Compute and report estimated USD cost |
| `FURROW_DEFAULT_COST_PER_1K` | `0.0,0.0` | Default per-1k token price for unknown models (USD) |
| `FURROW_MODEL_PRICING` | (none) | Mapping of `provider:model` to `input_per_1k,output_per_1k` USD prices |

A `.env` file is read automatically via pydantic-settings.

## LLM token accounting and cost

The LLM client records input/output tokens and call counts for every completion, and the orchestrator accumulates them per cycle and per run. Usage is persisted in the run state file and printed by the CLI in the run summary and in `furrow status`.

When `FURROW_TRACK_COST` is true (the default), Furrow also estimates the USD cost of every call using built-in pricing tables for common models. Unknown models fall back to a free default price (`FURROW_DEFAULT_COST_PER_1K`), and custom prices can be supplied via `FURROW_MODEL_PRICING`. Estimated cost is shown in the CLI summary/status and in the web UI header.

## State & resume

State is persisted atomically after every cycle to `<workspace>/.furrow/state.json`. Use `furrow resume` or `furrow start --resume` to continue a previous run. Corrupt state files are ignored and the run starts fresh. On resume, the persisted state file is loaded exactly once, so restored values cannot be overwritten by a second read.

## Development

```bash
cd /workspace/app
python -m pytest -q           # run the full offline test suite
python -m ruff check .        # linting
python -m mypy furrow         # type checking
```

All tests run offline and do not require API keys.