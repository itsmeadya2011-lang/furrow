# Furrow Agent Architecture

## Overview

Furrow is an autonomous coding agent that runs a plan -> execute -> test loop until the goal
is met or a limit is reached. Two implementations of that loop exist in this repository.

## Relationship between `.kilo/` and `furrow/`

They are separate codebases that share a goal and a task JSON shape. They do not import or
call each other.

- **`.kilo/`** — markdown agent definitions and a slash command, loaded by Kilo. The
  orchestrator agent (`mode: primary`) runs the loop using Kilo's own subagent tooling: it
  plans itself, spawns `furrow-worker` subagents via the Task tool, spawns `furrow-tester`,
  and keeps its own short internal todo list as cycle state. This side has no persistent
  state file and no code of its own.
- **`furrow/`** — the standalone Python implementation, and the substantive one: orchestrator,
  planner/worker/tester agents, provider-agnostic LLM client with sandboxed file tools, CLI,
  and a web UI. It runs without Kilo.

Both sides use the same plan schema (`tasks[]` with `id`, `description`, `files`,
`dependencies`, plus a `rationale`) and the same verdict schema (`passed`, `summary`,
`failures`), which is why the `.kilo/` agent definitions mirror the Python agent prompts.

## Kilo-side components

### 1. Orchestrator (`.kilo/agent/furrow.md`)
- **Type:** primary agent (`steps: 50`), also `default_agent` in `kilo.json`
- **Role:** Runs the loop: identify goal, plan 1-5 parallelizable tasks, spawn
  `furrow-worker` subagents, spawn `furrow-tester`, report, repeat. Instructed to run until
  the goal is complete or the user stops it.

### 2. Planner (`.kilo/agent/furrow-planner.md`)
- **Type:** subagent
- **Role:** Break a goal into 1-5 parallelizable tasks, each completable in 1-3 tool call
  rounds.
- **Output:** JSON `{tasks: [{id, description, files, dependencies}], rationale}`.

### 3. Worker (`.kilo/agent/furrow-worker.md`)
- **Type:** subagent
- **Role:** Execute one assigned task with minimal, targeted changes. Does not scope creep,
  does not spawn subagents.

### 4. Tester (`.kilo/agent/furrow-tester.md`)
- **Type:** subagent
- **Role:** Run the project's test suite, linter, and type checker; its definition
  instructs it to diagnose and fix failures, then return `{passed, summary, failures}`.
  (Contrast with the Python `TesterAgent` below, which only reports.)

### 5. Command (`.kilo/command/furrow.md`)
- **Type:** slash command, `agent: furrow`, `subtask: false`
- **Role:** Entry point. Starts the loop with the goal from `$ARGUMENTS`, asking the user for
  one if absent.

## Python components (`furrow/`)

### Orchestrator (`furrow/core/orchestrator.py`)
- **Type:** `Orchestrator(goal, client=None, settings=None)`
- **Role:** `await orchestrator.run()` executes the loop. Each cycle: plan, execute, test.
- **Execution:** tasks run in dependency-ordered waves. A task becomes runnable once every
  dependency it declares that exists in the plan is `completed`. At most
  `settings.max_parallel_tasks` workers run concurrently (asyncio semaphore). A dependency
  cycle is logged and its remaining tasks run together rather than deadlocking; tasks whose
  dependencies can never complete are marked `skipped`.
- **Feedback:** when the tester reports failure, the failure lines become the next cycle's
  `plan_context`. `self.goal` itself is never reassigned.
- **Cycle errors:** any exception raised by a cycle counts as a hard failure, and so does a
  cycle whose test run is red. Both are charged to the same `_consecutive_non_progress`
  streak that `fail_fast_cycles` bounds; the loop continues until the streak is exhausted.

### Planner (`furrow/agents/planner.py`)
- **Contract:** `await PlannerAgent.plan(goal, failure_context=None, **extra) -> Plan`
- **Role:** Sends a bounded listing of the workspace tree plus the goal (and any failure
  context) and asks for plan JSON. Response text goes through `extract_json` /
  `parse_plan`, which tolerate fenced blocks, a bare array, a `{...}` span, and a
  `{"1": {...}}` mapping. Malformed responses get one repair round-trip; a second failure
  raises `ValueError`. Unknown keys, hallucinated `status`/`result` values and non-object
  task entries are dropped; at most `MAX_TASKS` (10) tasks are kept.

### Worker (`furrow/agents/worker.py`)
- **Contract:** `await WorkerAgent(task=..., client=..., settings=...).run() -> WorkerResult`
- **Result:** `WorkerResult(success: bool, summary: str, files_changed: list[str],
  error: str | None)`
- **Role:** Implements one `TaskModel` through an LLM tool loop and reports what it changed.
  `run()` never raises — failures come back as `WorkerResult(success=False, error=...)` so
  one bad task cannot abort the whole cycle. A summary matching self-reported-inability
  phrasing ("I cannot complete...", "no changes were made") is treated as a failure.
- **Tools:** `read_file`, `write_file`, `edit_file`, `list_files`, all confined to
  `settings.workspace` by `LLMClient.resolve_path`; a path resolving outside it is refused.
  `edit_file` requires an exact `old_string` match and requires uniqueness unless
  `replace_all=true`. Tool errors are returned to the model as text so it can retry.
  Budget: `settings.max_tool_turns` rounds per task.

### Tester (`furrow/agents/tester.py`)
- **Contract:** `await TesterAgent.run(goal, tasks) -> TestResult`
- **Result:** `TestResult(passed: bool, summary: str, failures: list[str])`
- **Role:** Executes `settings.test_command` if set, otherwise probes for a runner
  (`pytest -q`, `python -m pytest -q`, `npm test -- --silent`, `pnpm test -- --silent`,
  `yarn test --silent`, `cargo test -q`, `go test ./...`) and runs the first one that is
  installed and actually executes. Timeout `settings.test_timeout`; a timed-out process is
  killed and reaped. The run's output is handed to the model, which returns a JSON verdict.
- **The process exit code is authoritative:** a non-zero exit or timeout is a failure even
  if the model claims success. If the model's reply is unparseable, the verdict falls back
  to the exit code, and a runner that never executed yields a failure result with a summary
  saying the goal could not be verified — which is not the same claim as "the code is
  broken".
- **It only reports.** `TesterAgent` performs no repairs; fixes come from the next cycle's
  plan. (The `.kilo/agent/furrow-tester.md` prompt is what tells the Kilo-side subagent to
  fix failures.)

## Loop termination rules

`Orchestrator.run()` halts when any of these hold:

1. **Goal complete** — every task planned this cycle is `completed`, none `failed`, and the
   test run did not fail. A cycle that planned nothing ran no tests, so it can complete the
   loop; a cycle whose tests failed never can, regardless of how clean the tasks finished.
2. **`max_cycles` reached** — `settings.max_cycles` (`0` = unlimited; the default). Note the
   Python loop is bounded by default only when you set `FURROW_MAX_CYCLES` or
   `--max-cycles`; `0` means unlimited.
3. **`fail_fast_cycles` consecutive cycles without progress** — `settings.fail_fast_cycles`
   (default 3) cycles in a row that each either raised an unexpected exception or finished
   with a failing test run. Non-positive values are clamped to 1, so `0` cannot switch the
   bound off. The streak resets on any cycle whose tests pass, and on a cycle that planned no
   tasks, so an empty plan cannot spin. The halt reason names the cause ("hard failure" vs
   "failing test run").

`KeyboardInterrupt` and `CancelledError` are not swallowed.

## Files

```
.kilo/
  agent/
    furrow.md          # Orchestrator (primary agent)
    furrow-planner.md  # Planner subagent
    furrow-worker.md   # Worker subagent
    furrow-tester.md   # Tester subagent
  command/
    furrow.md          # /furrow slash command
furrow/
  __init__.py
  config.py            # Provider, TaskStatus, TaskModel, Plan, TestResult, Settings
  llm.py               # LLMClient + sandboxed workspace file helpers
  core/
    orchestrator.py    # the plan/execute/test loop
  agents/
    planner.py
    worker.py
    tester.py
    prompts.py
  cli/
    main.py            # `furrow start`, `furrow web`, `furrow --version`
  web/
    server.py          # FastAPI: GET /, WebSocket /ws
tests/
  conftest.py
  test_config.py
  test_core.py
kilo.json              # Project config (default agent = furrow, open permissions)
pyproject.toml         # Packaging, ruff and mypy config
README.md              # User-facing docs: install, CLI, env vars, limitations
```

## Permissions

`kilo.json` sets `bash: allow`, `edit: allow`, `read: allow`.

## Usage

- Kilo side: run `/furrow <goal>`; the agent loops until the goal is complete or the user
  stops the session. Mid-loop user input is folded into the next cycle's plan.
- Python side: `furrow start "<goal>"` or `furrow web`. See `README.md` for flags and the
  `FURROW_`-prefixed environment variables.

## Open Items

- `Provider.OLLAMA` is declared in `furrow/config.py` and defaulted in the CLI's per-provider
  model table, but `LLMClient` raises a clear "not implemented" error. Either implement it or
  drop the enum value and the now-misleading defaults.
- The web server (`furrow/web/server.py`) has no authentication. It defaults to
  `127.0.0.1`, but nothing prevents binding it publicly.
- The web server's browser client renders structured `plan` / `task_start` / `task_done` /
  `test_result` / `cycle_end` frames, and `_QueueLogHandler` forwards any record logged with
  `extra={"event": {...}}` as such a frame — but nothing in `furrow/` logs an `event`
  attribute today. Those frames are never produced; the browser gets the `accepted`,
  `cycle_start` and `done`/`error` frames that the server queues itself, plus plain log
  lines.
- No persistent state: run state lives in the `Orchestrator` instance (and, on the Kilo side,
  in the agent's context). Resuming a run across processes or sessions is not supported.
- The tester detects a test runner; it does not run the linter or type checker as part of the
  loop, even though `ruff` and `mypy` are configured in `pyproject.toml`.
- `mypy` is configured but was not brought to a clean state.