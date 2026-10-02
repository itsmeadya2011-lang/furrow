# Furrow

Furrow is an autonomous coding agent with an infinite parallel development loop. It takes
a single goal, plans it into parallelizable tasks, runs those tasks concurrently through an
LLM, runs the project's test suite, and feeds the result back into the next cycle until the
goal is done, the cycle budget is exhausted, or there is nothing left to plan.

## Install

```bash
pip install furrow
```

## Quickstart

```bash
furrow start "Add JWT auth to the API"
furrow web
```

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment before starting a run. The
provider default is `anthropic`, and each SDK key is also readable through the
`FURROW_`-prefixed settings described below.

The console script is declared in `pyproject.toml` as `furrow = "furrow.cli.main:main"`.
The group has exactly two subcommands:

```text
Usage: furrow [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  start
  web
```

`start` takes an optional positional goal and an optional model override:

```text
Usage: furrow start [OPTIONS] [GOAL]

Options:
  --model TEXT  Override the LLM model for this run. Applies to the planner,
                worker and tester alike, so per-role models
                (FURROW_PLANNER_MODEL and friends) are ignored while this flag
                is set.
  --help        Show this message and exit.
```

If `GOAL` is omitted, `start` prompts for it.

`--model X` sets the `model`, `planner_model`, `worker_model` and `tester_model` fields on a
*copy* of the settings for that run only. The planner, worker and tester therefore all use
`X`, and the flag takes precedence over `FURROW_PLANNER_MODEL`, `FURROW_WORKER_MODEL` and
`FURROW_TESTER_MODEL`. The process-wide `furrow.config.settings` singleton is never mutated,
so concurrent `furrow web` runs cannot interfere with each other. Omit the flag to keep
per-role models from the environment.

```bash
furrow start "Add JWT auth to the API" --model claude-opus-4-20250514
```

`web` serves the browser console and the HTTP/WebSocket API described below.

## How the loop works

One cycle in `furrow/core/orchestrator.py` is:

1. **Plan** — `PlannerAgent.plan()` asks the LLM (with `planner_model`) to break the current
   directive into 1-5 parallelizable tasks. The plan is emitted as a `plan` event.
2. **Execute** — one `WorkerAgent` per task runs concurrently, bounded by
   `max_parallel_tasks`, each using `worker_model`. Each task emits `task_start` and
   `task_end`.
3. **Test** — `TesterAgent` runs the project's test runner with `tester_model`, emits
   `test_result`, and reports `passed`, `summary`, `failures`.
4. **Record** — the cycle's plan, tasks and test result are appended to `history` as an
   immutable `CycleRecord`, and a `cycle_end` event is emitted.
5. **Repeat** — the loop starts the next cycle.

### Halt conditions

The loop halts when **any** of these is true:

- **Every task in the most recent cycle is `completed`.** `_last_cycle_complete()` returns
  `True` only when history is non-empty, the last cycle's task list is non-empty, and all
  task statuses are `completed`.
- **The cycle budget is spent.** `_limit_reached()` is
  `max_cycles > 0 and cycles >= max_cycles`. With `max_cycles = 0` (the default) the budget
  is unlimited and this never triggers. If the last cycle also fully succeeded, the run
  finishes; otherwise it halts with `Reached the max_cycles limit (N). Halting.`
- **The planner returned no tasks.** `_cycle()` returns `None`, and the run halts with
  `Planner produced no tasks. Halting to avoid an unbounded loop.`

A cycle containing **any failed task is not done**, so the loop continues.

> **`completed` does not mean "tests passed".** A task's status is `completed` if and only
> if `WorkerAgent.run()` returned without raising. The test result is not consulted when
> deciding whether the last cycle was complete. A run can therefore reach
> `run_complete` even though the last `test_result` had `passed: false`. Do not read a
> completed cycle as evidence that the goal is satisfied.

### Goal versus directive

`Orchestrator.goal` is the immutable original goal. A test failure does not overwrite it.
Instead the loop sets `Orchestrator.directive` for the *next* cycle only:

- tests pass: `directive = goal`
- tests fail: `directive = "Fix failing tests:\n" + "\n".join(test_result.failures)`

The tester is always invoked against the original `goal`, never the mutated directive.

## Configuration

Settings are declared in `furrow/config.py` with
`SettingsConfigDict(env_prefix="FURROW_", env_file=".env")`, so every field is settable as a
`FURROW_`-prefixed environment variable or in a `.env` file.

| Field | Environment variable | Type | Default | Meaning |
| --- | --- | --- | --- | --- |
| `provider` | `FURROW_PROVIDER` | `Provider` | `anthropic` | `anthropic`, `openai` or `ollama` |
| `model` | `FURROW_MODEL` | `str` | `claude-sonnet-4-20250514` | Fallback model in `LLMClient.complete` |
| `planner_model` | `FURROW_PLANNER_MODEL` | `str` | `claude-3-5-haiku-20241022` | Model used by `PlannerAgent` |
| `worker_model` | `FURROW_WORKER_MODEL` | `str` | `claude-3-5-sonnet-20241022` | Model used by `WorkerAgent` |
| `tester_model` | `FURROW_TESTER_MODEL` | `str` | `claude-3-5-sonnet-20241022` | Model used by `TesterAgent` |
| `anthropic_api_key` | `FURROW_ANTHROPIC_API_KEY` | `str \| None` | `None` | Falls back to `ANTHROPIC_API_KEY` |
| `openai_api_key` | `FURROW_OPENAI_API_KEY` | `str \| None` | `None` | Falls back to `OPENAI_API_KEY` |
| `ollama_base_url` | `FURROW_OLLAMA_BASE_URL` | `str` | `http://localhost:11434` | `/v1` is appended by the client |
| `max_parallel_tasks` | `FURROW_MAX_PARALLEL_TASKS` | `int` | `5` | Worker concurrency cap and plan-size cap |
| `max_cycles` | `FURROW_MAX_CYCLES` | `int` | `0` | `0` means unlimited |
| `test_timeout` | `FURROW_TEST_TIMEOUT` | `int` | `300` | Seconds before the test run is killed |
| `workspace` | `FURROW_WORKSPACE` | `Path` | `Path.cwd()` | Working directory for the test command |
| `log_level` | `FURROW_LOG_LEVEL` | `str` | `INFO` | Threshold for structured logs on stderr |

`max_cycles = 0` means unlimited: `_limit_reached()` requires `max_cycles > 0` before it can
ever return `True`.

## Providers

`provider` accepts `anthropic`, `openai` and `ollama` (see the `Provider` enum in
`furrow/config.py`).

- **Anthropic** uses `AsyncAnthropic.messages.create` with `max_tokens=4096`.
- **OpenAI** uses `AsyncOpenAI.chat.completions.create`.
- **Ollama** goes through the OpenAI-compatible surface. `LLMClient.ollama` builds the
  client as `AsyncOpenAI(base_url=f"{base_url}/v1", api_key="ollama")`, where `base_url` is
  `ollama_base_url` with any trailing slash stripped. The key is a placeholder that Ollama
  ignores but the SDK requires.

All three completion paths are wrapped by a tenacity retry declared as:

```python
retry_transient = retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    reraise=True,
)
```

with `_MAX_ATTEMPTS = 3`. `_is_transient` is `True` for connection errors, timeouts, and
API errors whose `status_code` is `429` or greater than or equal to `500`. Configuration
errors are deliberately excluded: a missing key raises `ValueError("ANTHROPIC_API_KEY is not
set")` (or the `OPENAI_API_KEY` equivalent) from the client property, and that is never
retried because a retry cannot succeed.

## Web UI + API

```bash
furrow web
```

The server is `uvicorn.run(app, host="0.0.0.0", port=8000)`, so it listens on `0.0.0.0:8000`.
`python -m furrow.web.server` is an equivalent entry point.

### REST routes

| Method | Path | Result |
| --- | --- | --- |
| `GET` | `/` | The single-page HTML console |
| `POST` | `/api/runs` | `202` with a run snapshot; `422` if `goal` is blank |
| `GET` | `/api/runs` | Array of run snapshots |
| `GET` | `/api/runs/{run_id}` | One snapshot; `404` for an unknown `run_id` |

`POST /api/runs` takes `{"goal": "...", "model": "..."}` with `model` optional. Every route
returns the same snapshot, built as `{"run_id": run_id, **orchestrator.status()}`:

```json
{
  "run_id": "…",
  "goal": "…",
  "directive": "…",
  "cycles": 0,
  "running": true,
  "done": false,
  "tasks": [{"id": "1", "description": "…", "status": "completed", "result": "…"}],
  "last_test": {"passed": false, "summary": "…", "failures": ["…"]}
}
```

`tasks` describes the most recent cycle (empty before the first cycle completes) and
`last_test` is `null` until a test result exists. Runs live in a module-level registry in
`furrow/web/server.py` and are pumped by background `asyncio` tasks, so they are not tied to
the lifetime of the request or socket that created them. The server retains the 50 most
recently finished runs, so a long-lived process does not grow without bound.

The `model` field behaves exactly like `furrow start --model`: it sets the `model`,
`planner_model`, `worker_model` and `tester_model` fields on a *copy* of the settings for
that run, so the planner, worker and tester all use it and concurrent runs can still differ
from one another. The global settings singleton is never mutated.

### WebSocket protocol

Connect to `/ws`. The **first client frame** is either:

- `{"goal": "...", "model": "..."}` — start a new run, or
- `{"run_id": "..."}` — attach to an existing run and keep streaming it.

Anything else produces an `error` frame and the socket is closed with code `1008`.

The server replies with one control frame — `run_started` or `run_attached`, both of which
carry `run_id`, `message` and a `status` snapshot — and then streams the orchestrator's
events verbatim until the run finishes. Each run keeps a bounded 256-event replay buffer,
so a client attaching to an already-finished run still receives its history. Event frames
use the envelope:

```json
{"type": "…", "cycle": 3, "message": "…", "data": {}}
```

The `type` values emitted by the `Orchestrator` are:

| `type` | Emitted when |
| --- | --- |
| `cycle_start` | A cycle begins; `data` has `goal` and `directive` |
| `plan` | The planner returned; `data` has `rationale` and `tasks` |
| `task_start` | A worker begins; `data` has `id` and `description` |
| `task_end` | A worker finishes; `data` has `id`, `status`, `result` |
| `test_result` | The tester finished; `data` has `passed`, `summary`, `failures` |
| `cycle_end` | The cycle finished; `data` has `passed`, `completed`, `failed` |
| `run_complete` | The loop halted because the goal was reached |
| `run_halt` | The loop halted for another reason, was cancelled, or aborted on error |

Three additional control frames exist only in the web layer and never come from the
orchestrator: `run_started`, `run_attached` and `error`. The `error` frame keeps the event
envelope shape and also carries an `error` string alias of `message`.

Each subscriber gets a bounded queue. If a socket is too slow, that subscriber drops its
*oldest* frames, so terminal events are never lost.

## Development

Install the package in editable mode with the development extras:

```bash
pip install -e ".[dev]"
```

Run the tooling with the canonical module invocations:

```bash
python -m pytest
python -m ruff check .
python -m mypy furrow
```

`[tool.pytest.ini_options]`, `[tool.ruff]` and `[tool.mypy]` are all configured in
`pyproject.toml`. pytest runs in `asyncio_mode = "auto"` with `testpaths = ["tests"]`; ruff
targets `py310` at a 100-character line length, excludes `build`, and applies an explicit
rule selection so that `ruff check .` is unambiguous; mypy targets Python 3.10 and
**excludes `tests/`**, so type checking covers the `furrow` package only.

### Logging

Logs are structured, via `structlog`, and always go to **stderr**. Progress output from the
orchestrator and CLI is `rich` console output on stdout, so keeping the streams separate is
what stops a log line from corrupting the terminal or from breaking
`furrow start ... | tee`.

The threshold comes from `log_level`, i.e. `FURROW_LOG_LEVEL`, defaulting to `INFO`:

```bash
FURROW_LOG_LEVEL=DEBUG furrow start "Add JWT auth to the API"
```

An unrecognised level degrades to `INFO` rather than raising, so a typo cannot stop a run.
Logging is configured by the CLI at startup, once per command, not at import time —
importing `furrow` has no logging side effects.

## Project layout

```text
furrow/
  __init__.py          # lazy (PEP 562) re-exports of LLMClient and Settings
  config.py            # Provider, TaskModel, Plan, TestResult, Settings
  llm.py               # LLMClient: provider dispatch, retries, file helpers
  logging_config.py    # structlog setup on stderr, resolve_level, get_logger
  agents/
    __init__.py        # lazy re-exports of the three agent classes
    planner.py         # PlannerAgent plus plan normalisation (build_plan)
    worker.py          # WorkerAgent: one task, one completion
    tester.py          # TesterAgent: runs the test suite, judges the result
    prompts.py         # PLANNER_PROMPT, WORKER_PROMPT, TESTER_PROMPT
  core/
    __init__.py        # lazy re-export of Orchestrator
    orchestrator.py    # the plan/execute/test loop, worker pool, events
    json_utils.py      # extract_json: fenced and prose-wrapped JSON recovery
  web/
    __init__.py        # lazy re-exports of app and run
    server.py          # FastAPI routes, /ws streaming, HTML console
  cli/
    main.py            # click group: `start` and `web`
```

## Status

Early-stage project. The development loop, the CLI and the web console and API are
functional, but the loop is not yet capable of changing a codebase: the worker agents return
an LLM text summary for a task and do not apply file edits through a tool loop. Plans are
LLM-driven and therefore non-deterministic between runs, and no state is persisted across
process restarts. See [`docs/architecture.md`](docs/architecture.md) for the full list of
known limitations.

The repository does not currently declare a license.
