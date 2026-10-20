# Furrow

Furrow is an autonomous coding agent that runs a plan -> execute -> test loop over a goal. Each cycle a planner breaks the goal into independent tasks, up to N workers implement them in parallel by writing real files into the workspace, and a tester runs the project's test suite against the result. Failing tests are folded back into the next cycle, so the loop keeps working the goal until it is done, the cycle budget runs out, or the loop stops making progress.

## Install

```bash
pip install furrow      # released package
pip install -e .        # from a checkout
```

Furrow talks to an LLM provider directly. Set a key in your environment:

```bash
export ANTHROPIC_API_KEY=...   # Anthropic (default provider)
export OPENAI_API_KEY=...      # OpenAI
```

## Quick start

```bash
furrow start "Add JWT auth to the API"
```

The goal is optional; without one, `furrow start` prompts for it.

## Commands

| Command | Description | Options |
| --- | --- | --- |
| `furrow start [GOAL]` | Run the loop against a goal. | `--model`, `--max-cycles`, `--max-parallel`, `--state-file` |
| `furrow status` | Print the saved run: goal, cycles, task counts, last cycle result. | none |
| `furrow resume` | Continue the run stored in the state file. | `--model`, `--max-cycles`, `--max-parallel`, `--state-file` |
| `furrow web` | Serve the web UI and REST API on http://localhost:8000. | none |

`furrow start` and `furrow resume` write the run state to `--state-file` (default `.furrow/state.json`) after every cycle, so `furrow status` and `furrow resume` work across sessions.

## How it works

1. **Plan** — the planner turns the goal into 1-5 independent tasks with `id`, `description`, `files` and `dependencies`.
2. **Execute** — workers run in parallel, at most `max_parallel_tasks` at a time. Each worker reads the files its task declares plus a workspace listing, asks the model for a JSON `{summary, edits: [{path, content}]}` payload, and Furrow writes those files into the workspace. Edits that escape the workspace (absolute paths, `../`) are rejected.
3. **Test** — the tester picks the runner that matches the project (pytest, npm, cargo or go) and asks the model to judge the output.
4. **Repeat** — the orchestrator keeps every planned task and one history entry per cycle, and folds failing tests into the next cycle's goal.

The loop stops when the goal is complete (all planned tasks done and tests passing), when there is nothing left to plan, when `max_cycles` is reached, or when `max_consecutive_failures` cycles in a row complete no task.

## Configuration

Settings are read from `FURROW_`-prefixed environment variables or a `.env` file.

| Setting | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | `FURROW_PROVIDER` | `anthropic` | `anthropic`, `openai` or `ollama`. |
| `model` | `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model for all agents. |
| `planner_model` | `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the planner. |
| `worker_model` | `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the workers. |
| `tester_model` | `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the tester. |
| `anthropic_api_key` | `FURROW_ANTHROPIC_API_KEY` | unset | Falls back to `ANTHROPIC_API_KEY`. |
| `openai_api_key` | `FURROW_OPENAI_API_KEY` | unset | Falls back to `OPENAI_API_KEY`. |
| `ollama_base_url` | `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL of the Ollama server. |
| `max_parallel_tasks` | `FURROW_MAX_PARALLEL_TASKS` | `5` | Concurrent workers per cycle. |
| `max_cycles` | `FURROW_MAX_CYCLES` | `0` | Cycle budget; `0` means unlimited. |
| `max_consecutive_failures` | `FURROW_MAX_CONSECUTIVE_FAILURES` | `3` | Stop after this many cycles in a row complete no task. |
| `workspace` | `FURROW_WORKSPACE` | current directory | Root the workers may read and write. |
| `state_file` | `FURROW_STATE_FILE` | `.furrow/state.json` | Where run state is written. |
| `apply_worker_edits` | `FURROW_APPLY_WORKER_EDITS` | `true` | Set to `false` to stop workers writing files. |
| `log_level` | `FURROW_LOG_LEVEL` | `INFO` | Log level for the structured logs. |

## Web UI and REST API

`furrow web` serves a single page UI on http://localhost:8000 that streams the orchestrator's events over a websocket and renders a live event log and task list. The run continues server-side when the browser disconnects.

| Endpoint | Description |
| --- | --- |
| `GET /api/health` | `{"status": "ok", "running": bool}` |
| `GET /api/state` | Current goal, cycle count, tasks and per-cycle history. |
| `POST /api/start` | Start a run with `{"goal": "..."}`. `409` if a run is already in progress. |
| `POST /api/stop` | Cancel the running loop. |

The event stream carries `cycle_start`, `plan`, `task_start`, `task_done`, `task_failed`, `test_result`, `cycle_end` and `done`.

## Development

Requires Python 3.10 or newer.

```bash
pip install -e ".[dev]"

pytest -q
ruff check .
mypy furrow
```
