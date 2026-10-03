# Furrow

An autonomous coding agent. You give it a goal; it plans the work, executes the
tasks in parallel through an LLM, runs your project's test suite, feeds the
failures back into the next plan, and repeats until the goal is done — or until
it decides it is making no progress.

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [CLI reference](#cli-reference)
- [Environment variables](#environment-variables)
- [LLM providers](#llm-providers)
- [How the loop works](#how-the-loop-works)
- [Run state](#run-state)
- [Web UI](#web-ui)
- [Development](#development)

## Install

Furrow targets Python 3.10+. From a checkout:

```bash
pip install -e '.[dev]'     # runtime + pytest / ruff / mypy
pip install -e .            # runtime only
```

Installing puts a `furrow` executable on your `PATH` (entry point
`furrow.cli.main:main`).

## Quickstart

```bash
export ANTHROPIC_API_KEY=sk-...
cd ~/code/my-project

furrow start "Add JWT authentication to the API, with tests"
```

The loop runs until the goal is satisfied. Press `Ctrl+C` to stop it.

To watch a run in the browser instead of the terminal:

```bash
furrow web                  # http://localhost:8000
```

Both commands read configuration from `FURROW_*` environment variables (or a
`.env` file in the working directory) and take CLI flags that override it.

## CLI reference

`furrow --help` lists the commands; there are exactly two.

### `furrow start [GOAL]`

Runs the plan → execute → test loop on the terminal until it finishes or you
interrupt it. `GOAL` is optional — if omitted you are prompted for it.

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--provider` | `anthropic` \| `openai` \| `ollama` | the configured provider (`anthropic` unless overridden) | LLM provider. Case-insensitive. |
| `--model` | string | `claude-sonnet-4-20250514` | Override the model id used for completions. |
| `--max-cycles` | int | `0` | Stop after this many cycles. `0` means unlimited. |
| `--max-parallel` | int | `5` | Maximum number of tasks executed concurrently. Must be ≥ 1. |
| `--workspace` | path | current directory | The directory the agents are allowed to edit. Must already exist. |
| `--test-command` | string | auto-detected | Command that runs the test suite, run verbatim via `shlex`. |
| `--state-file` | path | `<workspace>/.furrow/state.json` | Where run state is persisted as JSON. |
| `-v`, `--debug` | flag | off | Print the full traceback when something fails. |

Furrow prints a plain-language error and exits `1` on a value it validates
itself — a blank goal, `--model`/`--test-command` left blank, `--max-cycles < 0`,
`--max-parallel < 1`, a workspace that is not an existing directory, a missing
API key, a failing run, and a run that ended because its planner kept failing
(the orchestrator raises `PlannerUnavailableError` there, so the exit status
agrees with the message). `-v` adds the traceback. Values rejected by click's
own types and choices (`--max-cycles abc`, `--provider bogus`, a `--port` outside
1–65535) are usage errors: click prints them and exits `2`. `Ctrl+C` prints a
short message and exits `0`; end-of-input at the goal prompt prints click's
`Aborted!` and exits `1`.

### `furrow web`

Serves the web UI (see [Web UI](#web-ui)).

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--host` | string | `0.0.0.0` | Interface to bind. |
| `--port` | int | `8000` | TCP port to bind. Must be 1–65535. |
| `-v`, `--debug` | flag | off | Print the full traceback when the server fails to start. |

`furrow start` builds a fresh `Settings` from the module-level singleton plus the
CLI overrides. `furrow web` takes no settings flags at all: the server builds a
fresh `Settings` per run, overriding only `model` when the start message carries
one. Neither mutates the singleton, so repeated in-process invocations never
inherit stale values.

## Environment variables

Every `Settings` field is settable as `FURROW_<FIELD>` (upper-cased), or in a
`.env` file in the working directory.

### Provider and model

| Variable | Default | Meaning |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | `anthropic`, `openai` or `ollama`. |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Model id for completions. |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the planner. |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by workers. |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the tester. |
| `FURROW_MAX_TOKENS` | `8192` | Per-completion token cap. |

### Credentials

| Variable | Default | Meaning |
| --- | --- | --- |
| `FURROW_ANTHROPIC_API_KEY` | unset | Anthropic key. Falls back to `ANTHROPIC_API_KEY`. |
| `FURROW_OPENAI_API_KEY` | unset | OpenAI key. Falls back to `OPENAI_API_KEY`. |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server base URL. |

### Loop control

| Variable | Default | Meaning |
| --- | --- | --- |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Concurrency cap for the task semaphore *and* the maximum number of tasks a plan may contain. |
| `FURROW_MAX_CYCLES` | `0` | Stop after N cycles; `0` means unlimited. |
| `FURROW_MAX_CONSECUTIVE_FAILURES` | `3` | Stop after this many cycles in a row fail identically. `0` disables the guard. It also bounds a planner that keeps failing — see [How the loop works](#how-the-loop-works). |
| `FURROW_MAX_TASK_ATTEMPTS` | `2` | Escalates a failing task to `skipped` once it has consumed this many attempts. See [How the loop works](#how-the-loop-works) — in the normal loop only `1` has any effect. |

### Testing and workspace

| Variable | Default | Meaning |
| --- | --- | --- |
| `FURROW_TEST_COMMAND` | unset | Explicit test command. When unset, the tester auto-detects one. |
| `FURROW_TEST_TIMEOUT` | `300` | Seconds allowed for the whole check phase. |
| `FURROW_WORKSPACE` | current directory | Directory the agents may edit. |
| `FURROW_STATE_FILE` | `<workspace>/.furrow/state.json` | Where run state is persisted. |
| `FURROW_LOG_LEVEL` | `INFO` | Declared log level. Not yet applied to a logging setup. |

## LLM providers

Set `FURROW_PROVIDER` or pass `--provider`.

### `anthropic` (default)

Needs `ANTHROPIC_API_KEY` (or `FURROW_ANTHROPIC_API_KEY`). Uses the native
Anthropic SDK and only extracts text from `TextBlock` content, so thinking,
tool-use and other non-text blocks are ignored rather than crashing the call.

### `openai`

Needs `OPENAI_API_KEY` (or `FURROW_OPENAI_API_KEY`). Uses the native OpenAI
SDK.

### `ollama`

Needs a running Ollama server — no API key. Point `FURROW_OLLAMA_BASE_URL` at it
if it is not on `http://localhost:11434`. Furrow talks to Ollama through its
OpenAI-compatible endpoint, so only the base URL differs from the `openai`
provider; a placeholder key is sent because the OpenAI SDK rejects an empty one
and Ollama ignores the value.

Transient failures (rate limits, connection/timeout errors, 5xx) get up to
`RETRY_ATTEMPTS` (3) attempts with exponential backoff — one call plus two
retries. Client errors — a bad API key, a bad request — are surfaced immediately,
because retrying them only wastes wall clock.

## How the loop works

Each cycle is **plan → execute → test**:

1. **Plan.** `PlannerAgent.plan(goal, context)` asks the model for JSON
   describing a set of tasks (`id`, `description`, `files`, `dependencies`).
   `context` carries the previous cycle's test failures forward, so the plan
   reacts to what actually broke. The response is parsed leniently
   (markdown fences, prose around the payload, double-encoded strings, bare
   strings where a list is expected), with one corrective retry before giving up.
   A plan longer than `max_parallel_tasks` is truncated to the first N tasks.
2. **Execute.** Tasks run in dependency waves: everything whose dependencies are
   complete starts together, bounded by an `asyncio.Semaphore` of
   `max_parallel_tasks`. Missing dependencies, self-dependencies and dependency
   cycles mark a task `blocked`, and blocking propagates transitively to its
   dependants; so does a dependency that ended `failed` or `skipped`, because
   work whose prerequisite died can never run either. No task is ever left
   `pending` at the end of a wave: anything the plan made undispachable is
   closed out `blocked` with an `error` naming the dependency and its status,
   while anything the *run* declined to start (a stop request, or a wave that
   changed no status) is closed out `skipped` with an explanatory `error`. A
   task that fails stays `failed` and is escalated to `skipped` once
   `task.attempts >= max_task_attempts`.

   **The attempt budget is mostly unreachable, and that is by design of the
   re-planning loop, not a bug in the counter.** `WorkerAgent.run` owns
   `attempts`, but the orchestrator re-plans every cycle and only dispatches
   `PENDING` tasks, so each cycle's tasks arrive with `attempts = 0` and a given
   task object is dispatched at most once. A failure therefore stays `failed` at
   the default `max_task_attempts = 2`; setting it to `1` escalates on the first
   failure. Retry-after-failure comes from the next cycle's plan, which is free
   to emit the same task again — at a fresh attempt count.
3. **Test.** `TesterAgent.run(goal, tasks)` runs the project's checks, then asks
   the model for a JSON verdict (`passed`, `summary`, `failures`). The failures
   become the next cycle's context.

On failure the **goal is never rewritten** — only the focus context is — and the
loop keeps going until one of these happens:

- every task is `completed` **and** the test verdict passed (an empty plan counts,
  as long as the planner actually returned it);
- `max_cycles` is reached (`0` = never);
- `max_consecutive_failures` cycles in a row failed identically (no-progress
  guard), or you called `stop()` / pressed `Ctrl+C`;
- the planner failed to produce a plan on `max_consecutive_failures` cycles in a
  row. A planner failure is reported as an `error` event and the cycle still runs
  the tester, but it yields no tasks, so it is never mistaken for a finished
  goal: `max_cycles = 0` does not make a broken planner an infinite run.

The test runner is auto-detected by probing `pytest -q`, `python -m pytest -q`,
`python3 -m pytest -q`, `npm test`, `pnpm test`, `yarn test`, `cargo test -q` and
`go test ./...` in that order, skipping any candidate that is missing or collected
nothing; `ruff check .` and `mypy furrow` are run after the suite and their output
is appended when the tools exist. A cycle in which nothing could be verified at
all — no runner found, or every candidate inconclusive — is reported as **passed**
with a summary saying nothing was verified, never as a failure. Note the
consequence: a project with no detectable test runner reports a successful run as
soon as its tasks complete. Pass `--test-command` to get a real verdict.

### Progress events

`Orchestrator(goal, client, on_event, settings)` takes an `on_event` callback
with the signature `(name: str, payload: dict) -> None`. It is called
synchronously from inside the loop and is best-effort: an exception raised by
your callback is swallowed and never breaks a run.

| Event | Payload |
| --- | --- |
| `cycle_start` | `{"cycle": int}` |
| `plan` | `{"rationale", "task_count", "tasks": [{id, description, files, dependencies, status}]}` |
| `task_start` | `{"id", "description"}` |
| `task_done` | `{"id", "status", "result", "error"}` |
| `test_result` | `{"passed", "summary", "failures"}` |
| `cycle_end` | `{"cycle", "passed", "completed": [ids], "failed": [ids]}` |
| `done` | `{"reason", "cycles"}` — the last event of a run that finishes cleanly |
| `error` | `{"message", "where"}` — planner/tester failure that was recovered from |

`done` carries a `reason` such as `All tasks completed and tests passed`,
`No progress after N identical failing cycles`, `Reached max_cycles (N)`,
`Planner failed to produce a plan for N consecutive cycles (last error: ...)`,
`Stop requested` or `Cancelled`. It is emitted for every exit the loop owns;
an unexpected exception escaping a cycle propagates out of `run()` without one.
The planner-failure exit is the one case where `run()` also raises
`PlannerUnavailableError`, after emitting `done`, because a caller that never
sees the event stream — the CLI passes no `on_event` — would otherwise report a
run that planned and ran nothing as a success.

## Run state

Every cycle's goal, task list and outcome are written atomically to
`<workspace>/.furrow/state.json` (override with `FURROW_STATE_FILE` or
`--state-file`). Reads and writes are best-effort: a missing or corrupt file
never crashes a run. The file is deleted once the goal is fully achieved.

The file is a `RunState`: the goal, cycle count, the full `TaskModel` list
(status, result, error and attempt count per task) and a `CycleRecord` history
bounded to the most recent `MAX_HISTORY_RECORDS` (50) cycles, oldest dropped
first, because `max_cycles = 0` means unbounded. It is plain JSON, so it is safe
to inspect, diff or archive between runs. It is written for observability and
crash forensics — `StateStore.load()` exists but the orchestrator does not yet
resume a run from it.

## Web UI

`furrow web` serves a single page at `/` that opens a websocket to `/ws`. The
page sends `{"goal": "..."}` to start a run and `{"action": "stop"}` to wind it
down; the server also accepts an optional `"model"` key in the start message,
which the page itself never sends. The server streams one JSON object per
orchestrator event, so a long run shows live progress instead of sitting on
"Starting...". The last message is `{"event": "run_finished", "status":
"completed"|"stopped"|"failed", "goal": ..., "detail": ...}`, where `"detail"` is
empty unless the run raised. `"completed"` means the run ended on its own terms,
not that the goal was achieved: read the `done` event's `reason` for that.

Only one run may be in flight at a time — a second start is refused with an
`error` message and close code `4001`. A client that disconnects mid-run loses the
event stream and the server asks that run to wind down: it gets 30 s to finish,
then it is cancelled. Either way the run slot is released, so the next client
starts normally.

## Development

```bash
pytest -q                 # tests (asyncio_mode = "auto", testpaths = tests)
ruff check .              # lint, line-length 100, target py310
mypy furrow tests         # type check, source and tests
```

Or, with the dev extra installed: `pytest`, `ruff` and `mypy` directly.
