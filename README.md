# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Overview

Furrow is an autonomous coding agent that turns a high-level goal into a
continuous stream of parallel coding work. It runs a planner/worker/tester
loop: a planner breaks the goal into independent tasks, workers execute them
concurrently, and a tester verifies the result before the next cycle begins.
The loop continues until the plan is exhausted or a cycle limit is reached.

Furrow ships as two complementary layers:

- A standalone Python package providing the `furrow` CLI, the `Orchestrator`
  core, and a FastAPI-based web UI for monitoring runs.
- A Kilo TUI slash command (`/furrow`) that drives the same loop through the
  Kilo agent manager, treating the planner/worker/tester as subagents.

## Architecture

```
furrow CLI  ──►  furrow.core.Orchestrator
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    Planner        Workers (N)      Tester
   (Plan)        (concurrent)    (pass / fail)
```

- **`furrow` (CLI)** — Click entry point in `furrow.cli.main`. Dispatches to
  `furrow.core.Orchestrator`.
- **Planner** — Decomposes the goal into a `Plan` (see `furrow/config.py`)
  containing parallel `TaskModel` entries with explicit `dependencies`.
- **Workers** — Execute assigned tasks concurrently, bounded by
  `max_parallel_tasks`. Each worker makes minimal, targeted changes.
- **Tester** — Runs the project's tests/lint and returns a `TestResult`
  (passed, summary, failures). Failures feed back into the next planning
  cycle.
- **Loop** — The orchestrator repeats plan → work → test until no tasks
  remain or `max_cycles` is reached (`0` = unlimited).

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow start "Refactor database layer" --max-cycles 5 --max-parallel-tasks 3
furrow start "Fix failing tests" --planner-model claude-3-5-sonnet-20241022
furrow web
python -m furrow start "goal"
```

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.

## Configuration

Furrow reads configuration from environment variables and an optional
`.env` file (see `furrow/config.py`). All settings are prefixed with
`FURROW_`.

Required (at least one provider):

- `ANTHROPIC_API_KEY` — enables Anthropic models.
- `OPENAI_API_KEY` — enables OpenAI models.
- `OLLAMA_BASE_URL` — overrides the default Ollama endpoint
  (`http://localhost:11434`).

Optional `FURROW_*` overrides:

| Variable | Description | Default |
|---|---|---|
| `FURROW_PROVIDER` | `anthropic`, `openai`, or `ollama` | `anthropic` |
| `FURROW_MODEL` | Default model for all roles | `claude-sonnet-4-20250514` |
| `FURROW_PLANNER_MODEL` | Model used by the planner | `claude-3-5-haiku-20241022` |
| `FURROW_WORKER_MODEL` | Model used by workers | `claude-3-5-sonnet-20241022` |
| `FURROW_TESTER_MODEL` | Model used by the tester | `claude-3-5-sonnet-20241022` |
| `FURROW_MAX_PARALLEL_TASKS` | Concurrent workers per cycle | `5` |
| `FURROW_MAX_CYCLES` | Max planning cycles (`0` = unlimited) | `0` |
| `FURROW_WORKSPACE` | Working directory | current directory |
| `FURROW_LOG_LEVEL` | Log level | `INFO` |

## CLI flags

Flags for `furrow start`:

- `--max-cycles INT` — Maximum planning cycles. `0` means unlimited.
- `--max-parallel-tasks INT` — Maximum workers that run concurrently in
  one cycle.
- `--planner-model TEXT` — Override the planner model.
- `--worker-model TEXT` — Override the worker model.
- `--tester-model TEXT` — Override the tester model.
- `--model TEXT` — Override the default model for all roles.

## Web UI

`furrow web` launches a FastAPI server that provides a browser interface
for inspecting the current goal, plan, tasks, worker outputs, and test
results. It is useful for following long-running runs and diagnosing
where a cycle stalled.

## Development setup

```bash
git clone https://github.com/furrow/furrow.git
cd furrow
pip install -e ".[dev]"
pytest
```

The `[dev]` extra pulls in `pytest`, `pytest-asyncio`, `ruff`, and `mypy`.

## Agent architecture

For details on the Kilo TUI agent layer — including the `/furrow` slash
command and how the planner/worker/tester are wired up as Kilo subagents
— see [`.kilo/FURROW.md`](.kilo/FURROW.md).

## Roadmap

- `furrow.core.StateStore` — append-only run history at `.furrow/state.jsonl`
  (implemented as a low-level primitive; higher-level resume is upcoming).
- Real Ollama provider (`FURROW_PROVIDER=ollama` currently raises).
- Worker file writes (today the worker returns prose; future versions will
  parse structured output and call `LLMClient.write_file`).
- Desktop GUI wrapper around the same orchestrator.

## License

MIT.