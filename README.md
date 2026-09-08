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

Furrow reads settings from environment variables (prefixed with `FURROW_`) or a `.env` file in the working directory.

### Environment variables / `.env` file

| Variable | Default | Description |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, or `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model used by workers |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the planner agent |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by worker agents |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the tester agent |
| `FURROW_ANTHROPIC_API_KEY` | — | Anthropic API key |
| `FURROW_OPENAI_API_KEY` | — | OpenAI API key |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL for the Ollama provider |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Max concurrent worker tasks per cycle |
| `FURROW_MAX_CYCLES` | `0` | Max cycles (0 = unlimited) |
| `FURROW_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `FURROW_STATE_FILE` | `furrow_state.json` | Path to the persistent state file |
| `FURROW_WORKSPACE` | current directory | Working directory for the agent |

### Example `.env`

```env
FURROW_PROVIDER=ollama
FURROW_MODEL=llama3.1
FURROW_OLLAMA_BASE_URL=http://localhost:11434
FURROW_MAX_PARALLEL_TASKS=3
FURROW_MAX_CYCLES=10
```

### LLM Providers

Furrow supports three LLM providers:

- **Anthropic** — Set `FURROW_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` (or `FURROW_ANTHROPIC_API_KEY`).
- **OpenAI** — Set `FURROW_PROVIDER=openai` and `OPENAI_API_KEY` (or `FURROW_OPENAI_API_KEY`).
- **Ollama** — Set `FURROW_PROVIDER=ollama` and `FURROW_OLLAMA_BASE_URL` (defaults to `http://localhost:11434`). No API key required; Ollama must be running locally with the desired model pulled (`ollama pull llama3.1`).

## Persistent State

Furrow persists cycle progress to `furrow_state.json` (configurable via `FURROW_STATE_FILE`). The file stores the current goal, cycle count, and task states, allowing runs to be resumed or inspected across sessions.

## Logging & Observability

Furrow uses `structlog` for structured logging. Logs are emitted at `INFO` level by default and include events such as `run_started`, `cycle_started`, `plan_created`, `task_completed`, `task_failed`, `tests_passed`, `tests_failed`, and `max_cycles_reached`.

Configure verbosity with `FURROW_LOG_LEVEL` (e.g. `DEBUG`).

## Web UI

Run `furrow web` to start a local web server (default `http://0.0.0.0:8000`). The web UI lets you enter a goal and streams live progress over a WebSocket connection (`ws://<host>/ws`), showing real-time output from the orchestrator loop.

## Settings Reference

All settings are defined in `furrow/config.py` and loaded via `pydantic-settings`. They can be overridden by CLI flags (e.g. `furrow start "goal" --model gpt-4o`), environment variables, or the `.env` file.