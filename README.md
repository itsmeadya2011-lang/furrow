# Furrow

Autonomous coding agent with an infinite parallel development loop.

Furrow takes one goal, breaks it into tasks, implements them in parallel, runs the
test suite, and starts the next cycle. It is not a single-shot code generator: the
loop keeps going, taking the failures from the previous cycle as its new input,
until one of the conditions in [When the loop halts](#when-the-loop-halts) is met.

## How it works

Each cycle is plan -> execute -> test:

```text
cycle N
  1. plan     PlannerAgent.plan(current_goal) -> Plan(tasks=[...], rationale)
  2. merge    fold this cycle's tasks into the accumulated list, keyed by task id
  3. execute  ready-queue scheduler over TaskModel.dependencies,
              capped at max_parallel_tasks concurrent WorkerAgents
              (each worker writes whole files through LLMClient.write_file)
  4. test     TesterAgent.run(original_goal, plan.tasks) -> TestResult
  5. adjust   pass -> reset the failure counter
              fail -> current_goal becomes
                      "Original goal: <original>\n\nFix these failing tests:\n<failures>"
```

Cycle state lives in the `Orchestrator` object: `cycles`, `consecutive_failures`,
`tasks_planned_last_cycle` and the accumulated `tasks` list. A passing cycle does
not end the run -- that is the point of Furrow.

## Install

Furrow requires Python 3.10 or newer (`requires-python = ">=3.10"`).

```bash
pip install furrow
```

For a checkout, install in editable mode with the `dev` extra (pytest,
pytest-asyncio, ruff, mypy):

```bash
pip install -e ".[dev]"
```

Set a provider credential before running. `LLMClient` reads `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY` from the environment when the matching setting is unset; see
[Configuration](#configuration).

## Usage

### `furrow start`

```bash
furrow start "Add JWT auth to the API"
furrow start                                        # prompts for the goal
furrow start "Add JWT auth" --max-cycles 5 --max-parallel-tasks 2
```

The goal argument is optional. With no goal, `start` prompts for one
(`Enter your goal for Furrow`) before the loop begins.

| Option | Type | Effect |
| --- | --- | --- |
| `--model` | text | Overrides `Settings.model` for this run only |
| `--provider` | `anthropic`, `openai`, `ollama` | Overrides `Settings.provider` for this run only |
| `--max-cycles` | int | Cycle budget; `0` means unlimited |
| `--max-parallel-tasks` | int | Cap on tasks executed concurrently within a cycle |

Every option defaults to *unset*, and an unset option leaves the configured value
alone -- so omitting `--max-cycles` means the `max_cycles` setting applies, whose
default is `0`, i.e. unlimited. Overrides are applied to a per-invocation copy of
the settings; the process-wide `furrow.config.settings` singleton is never
mutated. `Ctrl-C` prints `Furrow stopped by user.` and exits 0.

### `furrow web`

```bash
furrow web
```

Launches the web UI on `http://0.0.0.0:8000` (see [Web UI](#web-ui)).

## Web UI

`furrow/web/server.py` exposes a FastAPI app titled `Furrow`:

- `GET /` serves a single-page UI: a goal field, an optional model field, and a
  start button.
- `ws://<host>/ws` is the event stream. On connect the page sends
  `{"goal": ..., "model": ... | null}`, and the server runs one `Orchestrator` on
  that connection, streaming one JSON event per websocket message. Exactly one
  terminal frame is always sent: `{"type": "done", "cycles": N}` or
  `{"type": "error", "message": "..."}`.

Frames are bounded before they go on the wire (`MAX_EVENT_BYTES = 8192` per frame,
`MAX_FIELD_CHARS = 2000`, `MAX_LIST_ITEMS = 50`, `MAX_FIELDS = 40`); an oversized
event is replaced by a `truncated: true` envelope. The bridge queue holds
`EVENT_QUEUE_SIZE = 256` frames, and a slow reader drops frames rather than
stalling the run. A per-connection model override is applied to a copy of the
settings, so one browser tab cannot leak a model into the CLI.

```python
from furrow.web.server import run

run(host="0.0.0.0", port=8000)
```

## Configuration

`Settings` in `furrow/config.py` is a `pydantic-settings` model. Every field
reads an environment variable with the `FURROW_` prefix, and a `.env` file in the
working directory is loaded as well.

| Field / env var | Default | Meaning |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | `anthropic`, `openai` or `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model passed to `LLMClient.complete()` |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the planner |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the worker |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the tester |
| `FURROW_ANTHROPIC_API_KEY` | unset | Falls back to `ANTHROPIC_API_KEY` |
| `FURROW_OPENAI_API_KEY` | unset | Falls back to `OPENAI_API_KEY` |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint; Furrow posts to `<base>/api/generate` |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Concurrent workers per cycle |
| `FURROW_MAX_CYCLES` | `0` | Cycle budget; `0` means unlimited |
| `FURROW_MAX_CONSECUTIVE_FAILURES` | `3` | Consecutive failing cycles before halting; `0` disables the limit |
| `FURROW_WORKSPACE` | `Path.cwd()` | Working directory for the test runner |
| `FURROW_LOG_LEVEL` | `INFO` | Declared, not yet read by any code |

Provider notes:

- `anthropic` and `openai` use their official SDKs and require an API key from
  either the `FURROW_*_API_KEY` setting or the plain `ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY` environment variable.
- `ollama` needs no key and talks HTTP to `ollama_base_url`.
- Transient failures (transport errors, timeouts, HTTP 408/409/429/500/502/503/504)
  are retried 3 times with exponential backoff between 1s and 10s. 4xx auth and
  validation errors are not retried.
- The planner, worker and tester each pass their own per-role model, so
  `FURROW_MODEL` is the fallback in `LLMClient.complete()` rather than the model
  the agents actually call with.

`FURROW_WORKSPACE` is where the tester runs the suite. The worker resolves the
relative paths it is given against the process working directory, so run Furrow
from the repository root.

Example `.env`:

```bash
FURROW_PROVIDER=anthropic
FURROW_MODEL=claude-sonnet-4-20250514
FURROW_PLANNER_MODEL=claude-3-5-haiku-20241022
FURROW_WORKER_MODEL=claude-3-5-sonnet-20241022
FURROW_TESTER_MODEL=claude-3-5-sonnet-20241022
FURROW_ANTHROPIC_API_KEY=sk-ant-...
FURROW_MAX_PARALLEL_TASKS=5
FURROW_MAX_CYCLES=0
FURROW_MAX_CONSECUTIVE_FAILURES=3
FURROW_WORKSPACE=/srv/projects/api
FURROW_LOG_LEVEL=INFO
```

Local models instead:

```bash
FURROW_PROVIDER=ollama
FURROW_OLLAMA_BASE_URL=http://localhost:11434
FURROW_PLANNER_MODEL=llama3.1
FURROW_WORKER_MODEL=llama3.1
FURROW_TESTER_MODEL=llama3.1
```

## When the loop halts

`Orchestrator.run()` checks three conditions after every cycle, in this order. It
halts on the first one that holds, emits a `halted` event with the reason, and
returns:

1. **The planner has nothing left to plan.** `_is_done()` is true when no tasks
   have been planned at all, or when the planner returned zero tasks this cycle.
   Reason: `goal complete: the planner has no work left to plan`.
2. **`max_consecutive_failures` consecutive cycles failed.** The counter is reset
   to zero by any passing cycle, so a run that alternates pass/fail never trips
   it, and `0` disables the check entirely. Reason:
   `failure limit reached: N consecutive cycles with failing tests`.
3. **The `max_cycles` budget is exhausted.** `0` means unlimited, so this only
   applies to a positive budget. Reason: `cycle budget reached: N/M cycles`.

The loop does **not** stop because a cycle succeeded, and a worker failure does
not end the run either: only that task is marked `failed` and the rest of the
cycle carries on.

## Task dependencies

The planner is asked for a `dependencies` list of task ids on every task, and
`TaskModel.dependencies` is honoured by the scheduler in
`Orchestrator._execute`:

- A task runs only after every task it depends on has **completed
  successfully**. Tasks with no dependencies between them still run in parallel,
  up to `max_parallel_tasks`.
- Dependencies are resolved against the current cycle's plan only. A reference to
  a task that is not in this plan is reported as a `dependency_warning` with
  reason `unknown_dependency` and ignored, rather than waited on forever.
- Cycles in the graph are found up front with Kahn's algorithm. The affected
  tasks are reported as a `dependency_warning` with reason `cycle` and marked
  `skipped` -- the run does not spin.

A `skipped` task is deliberately neither `completed` nor `failed`: nothing was
attempted. Tasks left over at the end of a cycle, whose dependency failed or was
skipped, are also marked `skipped` with an `unmet dependency` reason, so the
result set never claims work that did not happen.

The dependency graph is load-bearing, not decoration. Workers overwrite whole
files, so two workers writing the same file concurrently would silently clobber
each other; the graph is what keeps them apart.

## Events

`Orchestrator.on_event` is an optional synchronous hook called with a plain dict
for each lifecycle step. The orchestrator stamps `type` and `cycle` onto every
event and swallows exceptions raised by the hook, so a broken observer cannot
abort a run. The web UI is built on exactly this hook.

| `type` | Payload |
| --- | --- |
| `cycle_start` | `goal`, `focus`, `tasks_tracked` |
| `plan` | `task_ids`, `count`, `rationale`, `tasks_tracked` |
| `task_completed` | `id`, `status` (`completed`, `failed` or `skipped`), `result` |
| `test_result` | `passed`, `summary`, `failures`, `consecutive_failures` |
| `cycle_end` | `goal`, `tasks_completed`, `tasks_failed` |
| `halted` | `reason`, `cycles`, `consecutive_failures` |
| `dependency_warning` | `reason` (`cycle` or `unknown_dependency`), `message`, `refs` or `tasks` |

## Development

Run the test suite with pytest:

```bash
python3 -m pytest -q
```

`asyncio_mode = "auto"` is set in `pyproject.toml`, so `async def test_*` functions
need no `@pytest.mark.asyncio` marker. `testpaths` is `tests`.

Lint and format with ruff (`line-length = 100`, `target-version = "py310"`):

```bash
python3 -m ruff check .
python3 -m ruff format .
```

mypy is declared in the `dev` extra but has no configuration in `pyproject.toml`.
