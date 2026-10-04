# Furrow

An autonomous agent that plans, runs and tests work in an unbounded loop. It takes a
goal, asks a planner to break it into tasks, executes those tasks in dependency-ordered
parallel batches, runs your test suite, and feeds the failure output back into the next
plan until the goal is met or the loop stops itself.

## Current capability

**Furrow plans, reports and tests. It does not yet apply changes.**

`WorkerAgent` has no tool access. It cannot read, write or edit files, and there is no
diff application or file mutation anywhere in the worker path. A worker call is a single
LLM completion that receives the task description, the files it is told to touch, the
absolute workspace path and any prior failure output, and it returns a text report
describing what needs to change. A run therefore produces plans, worker reports and test
verdicts — not a modified repository.

The loop itself is real: parallel dependency-ordered execution, retries, failure feedback,
termination conditions and a live event stream are all implemented. See
[Architecture](#architecture) and [Current limitations](#current-limitations).

## Install

Install from a clone. `pip install furrow` is not a supported install path; there is no
release or publication configuration in this repository.

```bash
git clone <this repository>
cd furrow
pip install -e ".[dev]"
```

Python 3.10 or newer. The `dev` extra installs `pytest`, `pytest-asyncio`, `ruff` and
`mypy`.

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or see Configuration for the OpenAI path
furrow start "Add JWT auth to the API"
```

The loop runs until it decides it is done. Press Ctrl-C to stop it.

To watch a run in the browser instead:

```bash
furrow web     # http://127.0.0.1:8000
```

### Commands

| Command | Options |
| --- | --- |
| `furrow start [GOAL]` | `--model`, `--provider`, `--workspace`, `--max-cycles` |
| `furrow web` | `--host`, `--port` |
| `furrow --version` | — |

`GOAL` is optional; `furrow start` prompts for it if omitted. `--model` sets `model`,
`planner_model`, `worker_model` and `tester_model` together. `--workspace` must be an
existing directory. `--max-cycles 0` (the default) means no limit.

`python -m furrow` and `python -m furrow.cli` are equivalent to the `furrow` script.

## Configuration

Every setting is read from the environment with the `FURROW_` prefix, or from a `.env`
file in the working directory. Unrelated keys in a `.env` are ignored.

| Environment variable | Default | Notes |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | `anthropic` or `openai`. There is no Ollama backend; `ollama` is rejected at startup. |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Fallback model id. |
| `FURROW_PLANNER_MODEL` | `claude-sonnet-4-20250514` | |
| `FURROW_WORKER_MODEL` | `claude-sonnet-4-20250514` | |
| `FURROW_TESTER_MODEL` | `claude-sonnet-4-20250514` | |
| `FURROW_ANTHROPIC_API_KEY` | unset | Falls back to the bare `ANTHROPIC_API_KEY`. |
| `FURROW_OPENAI_API_KEY` | unset | Falls back to the bare `OPENAI_API_KEY`. |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Concurrency cap. Must be at least 1. |
| `FURROW_MAX_CYCLES` | `0` | `0` means unlimited. Must be 0 or more. |
| `FURROW_WORKSPACE` | current directory | Root the test suite runs in. |
| `FURROW_LOG_LEVEL` | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `FURROW_LLM_TIMEOUT` | `120.0` | Seconds per LLM request. |
| `FURROW_LLM_MAX_RETRIES` | `3` | Retries for transient provider errors. |
| `FURROW_LLM_MAX_TOKENS` | `8192` | Max output tokens per request. |
| `FURROW_TEST_TIMEOUT` | `300.0` | Seconds for the whole test-runner sweep. |
| `FURROW_TEST_COMMAND` | unset | Overrides auto-detection. Split with `shlex`, never run through a shell. |
| `FURROW_WEB_HOST` | `127.0.0.1` | |
| `FURROW_WEB_PORT` | `8000` | |

`FURROW_LOG_LEVEL` is honoured at CLI startup and configures stdlib logging on stderr.
There is deliberately no `--log-level` flag.

### API keys

Importing `furrow` and constructing `Settings` never requires a key. A key is only needed
at the point a provider call is made, and the error names the missing variable.

**`FURROW_PROVIDER=openai` is required to use the OpenAI key.** With the default
provider, setting only `OPENAI_API_KEY` fails with `ANTHROPIC_API_KEY is not set`, because
the Anthropic backend is the one selected.

```bash
export FURROW_PROVIDER=openai
export OPENAI_API_KEY=sk-...        # or FURROW_OPENAI_API_KEY
```

Validation runs when settings are constructed: `max_parallel_tasks >= 1`, `max_cycles >= 0`,
`log_level` constrained to the five levels above, non-blank model ids, non-blank
`test_command`, and `web_port` between 1 and 65535.

## Architecture

```
goal ──► plan ──► topological layers of worker tasks ──► run test suite ──► feedback
             ▲                                                                  │
             └──────────────────────────────────────────────────────────────────┘
```

| Module | Responsibility |
| --- | --- |
| `furrow/core/orchestrator.py` | Owns the loop: scheduling, task state across cycles, termination decisions, event emission. No provider logic. |
| `furrow/agents/planner.py` | Produces a `Plan` from the goal plus prior-cycle feedback. |
| `furrow/agents/worker.py` | One LLM call per task; returns a text report. No tools. |
| `furrow/agents/tester.py` | Runs the project's test suite and grades the run with the tester model. |
| `furrow/agents/prompts.py` | The three system prompts. |
| `furrow/llm.py` | Provider transport, retries, error taxonomy. |
| `furrow/tools/fs.py` | Workspace-confined filesystem helpers. Not yet wired into the worker. |
| `furrow/web/server.py` | FastAPI app: static UI, health probe, event websocket. |
| `furrow/config.py` | `Settings`, `Plan`, `TaskModel`, `TestResult`, `Provider`. |

### How a cycle works

1. **Plan.** The planner returns a JSON `Plan` of tasks, each with `id`, `description`,
   `files` and `dependencies`. Duplicate ids and dependencies on unknown ids are rejected
   at parse time. The prompt asks for 1–5 tasks; nothing enforces that. A task whose
   dependency is not in the same plan refers to work from an earlier cycle.
2. **Schedule.** Tasks run in topological layers: a task starts only once its dependencies
   have completed. Concurrency is capped by `max_parallel_tasks`. A task whose dependency
   failed or was never attempted never starts. A dependency cycle inside one plan is
   reported as a plan error rather than deadlocking the scheduler. A task that failed is
   re-queued at the start of a later cycle, for at most 2 attempts in total.
3. **Test.** The suite runs in `workspace` and the tester model grades the result as
   pass/fail with per-failure detail.
4. **Feedback.** Test failures, task failures, blocked tasks and plan errors are queued
   and attached to the next cycle's planner and worker calls. The goal itself is never
   overwritten by failure text. There is no interactive input channel.

### When the loop stops

Reported as **complete** (`run_done` carries `complete: true`):

- every task of the cycle reached `completed` and the tests passed;
- the planner returned no tasks and the tests passed; or
- the tests passed while failed tasks are past the attempt cap, so another cycle could not
  change the verdict. Reported explicitly, naming the tasks that did not complete.

Reported as **not complete** (`complete: false`, with a `reason`):

- `cancel()` was called, or Ctrl-C was pressed;
- `max_cycles` was reached;
- the circuit breaker tripped — 3 consecutive cycles raised; or
- stagnation — the planner returned no tasks 3 cycles running while the tests were still
  failing.

A run never reports the goal complete on a failure path. A single failed cycle does not
end the run; it is recorded, printed, and the loop continues.

### Event stream

The orchestrator reports everything twice: once to the console and once to an `on_event`
callback. The CLI and the web UI therefore observe exactly the same run. Every event is a
JSON object with a `type` and a human-readable `message`.

`run_start`, `cycle_start`, `plan`, `task_start`, `task_done`, `task_failed`, `tests`,
`cycle_done`, `warning`, `error`, `run_done`.

`run_done` additionally carries `complete`, `cancelled`, `cycles` and `reason`.

### Web UI

| Route | Purpose |
| --- | --- |
| `GET /` | The single-page UI. |
| `GET /health` | Liveness probe; returns status, version and whether a run is active. |
| `WS /ws` | Accepts `{"goal": "..."}` (1-4000 characters) and streams the run's events back as JSON. |

The server binds loopback only by default (`127.0.0.1:8000`). A WebSocket handshake
carrying a foreign `Origin` header is refused before it is accepted; a missing `Origin` is
treated as a non-browser client. One run at a time — a second concurrent start is
rejected. Closing the tab cancels the run cooperatively.

There is no authentication. Binding a non-loopback host is allowed but logs a warning:
anything that can reach the port can spend the configured API key.

## Reliability

- Transient provider failures (rate limits, timeouts, connection errors, overloaded, 5xx)
  are retried up to `llm_max_retries` times with exponential backoff. Each attempt carries
  its own `llm_timeout` budget. Auth failures and bad requests are not retried.
- Truncated answers are detected (`stop_reason: max_tokens`, `finish_reason: length`) and
  raise, rather than being returned as a complete answer.
- A provider reply with no usable text raises rather than surfacing later as a JSON parse
  error elsewhere.
- Malformed planner JSON is repaired by re-prompting with the parse error and the
  offending response, up to `llm_max_retries` times.
- The test runner is auto-detected from project marker files: `pyproject.toml`,
  `pytest.ini`, `setup.cfg` or `tox.ini` selects pytest; `package.json` selects npm, pnpm
  or yarn depending on the lockfile; `Cargo.toml` selects cargo; `go.mod` selects `go
  test ./...`. A runner that collects no tests is rejected in favour of the next candidate.
  `test_command` overrides detection entirely. The whole sweep shares the `test_timeout`
  budget, and captured output is trimmed to its last 8000 characters before it enters a
  prompt.
- Tests run with `cwd=workspace`, so a suite cannot be pointed outside it.
- A failed cycle, a failed test run and an unparseable tester response are recorded and
  fed back into the next plan instead of ending the run.

## Filesystem access

`furrow/tools/fs.py` confines all path handling to the configured workspace.
`resolve_in_workspace` rejects `..` escapes, absolute paths outside the workspace and
symlinks that point outside it, before any I/O happens. Listings skip `__pycache__`,
`.git`, `node_modules`, `.venv`, tool caches and build output, skip dotfiles, never follow
symlinks, and are capped in count and depth. Writes are atomic: content lands in a
temporary sibling and is moved into place, so an interrupted write cannot truncate a
source file.

This module is the foundation for the planned edit tool. It is not yet called from the
worker path.

## Verification

```bash
pytest
ruff check
ruff format --check
mypy
```

## Current limitations

- **The worker cannot change files.** No edit tool, no diff application, no file mutation.
  Worker output is a specification of what to change, not a record of a change.
- **`furrow/tools/fs.py` is not wired into the worker.** It is written and sandboxed, and
  it is the natural next step: give the worker `read`, `write` and `list` tools on top of it
  and the loop starts editing the workspace.
- **No persistent state.** Task statuses, attempt counts and the last test result live on
  the `Orchestrator` instance and are lost when the process exits. Nothing is written to
  disk and a restarted run starts from the goal alone.
- **No cost or token accounting.** There is no record of tokens, spend or per-task cost.
- **The web UI serves one run at a time**, unauthenticated, on loopback.
- **Three loop constants are not configurable**: 2 attempts per task, 3 consecutive errored
  cycles before the circuit breaker, and 3 consecutive empty plans before stagnation.
- `furrow --help` still opens with "an autonomous coding agent that plans, builds and tests
  in parallel". It plans and tests; it does not build yet.

## Relationship to the Kilo agents

This repository contains two independent implementations of the same idea:

- the `furrow/` Python package, which is what this README documents; and
- the Kilo agents under `.kilo/` (`furrow`, `furrow-planner`, `furrow-worker`,
  `furrow-tester`) plus the `/furrow` slash command, which drive the same plan/execute/test
  loop inside the Kilo agent runtime.

They are not wired to each other. The Python orchestrator does not call the Kilo agents,
and `/furrow` does not launch the Python loop. `.kilo/FURROW.md` documents the Kilo side;
`furrow start` and `furrow web` run the Python side.