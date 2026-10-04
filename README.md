# Furrow

Autonomous coding agent with an infinite parallel development loop.

Furrow takes a goal and runs a repeating loop: the planner breaks the goal into
1-5 parallelizable tasks, the workers execute them concurrently (bounded by a
parallelism cap), the tester runs tests, lint, and type checks, and the loop
repeats — folding failing-check output into the next cycle's plan — until the
goal is complete, the cycle budget is exhausted, or a stop is requested.

## Install

```bash
pip install furrow
```

This installs the `furrow` console script.

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow web
```

### `furrow start`

Run the loop on a goal until it is complete (or the budget/stopper intervenes).

```text
furrow start "GOAL" [OPTIONS]
```

Options:

| Option | Help text | Allowed values / type |
|--------|-----------|----------------------|
| `GOAL` (argument) | The objective to work on. | Required (or prompted if omitted) |
| `--model` | Override the LLM model. | string |
| `--provider` | Override the LLM provider. | `anthropic`, `openai`, `ollama` |
| `--max-cycles` | Maximum cycles (0 = unlimited). | integer |
| `--max-parallel` | Maximum parallel tasks. | integer |
| `--version` | Show the version and exit. | — |

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | Run completed successfully. |
| 1 | Run did not complete successfully. |
| 130 | Interrupted by the user (Ctrl+C). |

### `furrow web`

Start the web UI (served by uvicorn).

```text
furrow web [OPTIONS]
```

Options:

| Option | Help text | Default |
|--------|-----------|---------|
| `--host` | Interface to bind. | `0.0.0.0` |
| `--port` | Port to bind. | `8000` |
| `--log-level` | Uvicorn log level (e.g. debug). | — |

## Configuration

All settings are read from environment variables with the `FURROW_` prefix.
A `.env` file in the working directory is also read automatically.

| Environment variable | Type | Default | Notes |
|----------------------|------|---------|-------|
| `FURROW_PROVIDER` | string (`anthropic` / `openai` / `ollama`) | `anthropic` | LLM provider backend. |
| `FURROW_MODEL` | string | `claude-sonnet-4-20250514` | Default model. |
| `FURROW_PLANNER_MODEL` | string | `claude-3-5-haiku-20241022` | Model used by the planner. |
| `FURROW_WORKER_MODEL` | string | `claude-3-5-sonnet-20241022` | Model used by workers. |
| `FURROW_TESTER_MODEL` | string | `claude-3-5-sonnet-20241022` | Model used by the tester. |
| `FURROW_ANTHROPIC_API_KEY` | string | *(none)* | Required when provider is `anthropic`. |
| `FURROW_OPENAI_API_KEY` | string | *(none)* | Required when provider is `openai`. |
| `FURROW_OLLAMA_BASE_URL` | string (URL) | `http://localhost:11434` | Ollama endpoint (no key required). |
| `FURROW_MAX_PARALLEL_TASKS` | integer | `5` | Upper bound on concurrent tasks. |
| `FURROW_MAX_CYCLES` | integer | `0` | `0` means unlimited cycles. |
| `FURROW_WORKSPACE` | path | current directory | Directory whose project markers select checks. |
| `FURROW_LOG_LEVEL` | string | `INFO` | Log verbosity. |

### Providers

| Provider | API key required | Source |
|----------|-----------------|--------|
| `anthropic` | Yes | `FURROW_ANTHROPIC_API_KEY` or `ANTHROPIC_API_KEY` |
| `openai` | Yes | `FURROW_OPENAI_API_KEY` or `OPENAI_API_KEY` |
| `ollama` | No | Uses `FURROW_OLLAMA_BASE_URL` |

## Web UI

`furrow web` serves a single-page interface backed by FastAPI + uvicorn.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | HTML console with goal input, stop button, and a live event log. |
| `/api/status` | GET | JSON status: `running`, `goal`, `cycles`, task counts, and stop reason. |
| `/healthz` | GET | Liveness probe returning `{"status": "ok"}`. |
| `/ws` | WS | WebSocket streaming live plan, task, and test events. |

The websocket accepts two client messages:

- `{"goal": "...", "model": "..."}` — start a run.
- `{"action": "stop"}` — request the running orchestrator to stop.

The server pushes one JSON frame per orchestrator event, shaped as
`{"kind": "<kind>", "cycle": <n>, ...}`. Event kinds include `run_start`,
`plan`, `cycle_start`, `cycle_end`, `task_start`, `task_end`,
`test_result`, `stop_requested`, `stop_acknowledged`, `run_end`,
`dropped`, and `error`. Slow clients are non-blocking: events are dropped
(a `dropped` marker frame is emitted) rather than stalling the run.

Only one run can be active per process; a second start request while a run is
in progress is rejected with an error frame.

## Kilo agent architecture

Furrow also runs as a set of Kilo agents. `/furrow <goal>` invokes the `furrow`
orchestrator agent, which runs the same plan → parallel workers → test → repeat
loop through the agent definitions under `.kilo/agent/`:

| Agent | Role |
|-------|------|
| `furrow` | Primary orchestrator agent. |
| `furrow-planner` | Breaks a goal into parallelizable tasks. |
| `furrow-worker` | Executes a single assigned task. |
| `furrow-tester` | Runs tests, lint, and type checks. |

`.kilo/command/furrow.md` defines the `/furrow` slash command that starts the
loop with a goal from `$ARGUMENTS`. `kilo.json` sets the default agent to
`furrow` with open permissions (`bash: allow`, `edit: allow`, `read: allow`).

## Development

```bash
pip install -e ".[dev]"
python3 -m pytest -q
python3 -m ruff check .
python3 -m mypy furrow
```
