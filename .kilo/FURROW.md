# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs a plan -> execute -> test loop for
web/CLI/desktop projects.

There are two front ends onto the same loop:

- the Python package under `furrow/`, driven by `furrow start` or the web UI, and
- the Kilo agent definitions under `.kilo/agent/`, driven by the `/furrow` slash
  command.

This document describes the Python implementation first, because that is where the
loop semantics live; the agent files are prompts layered on top of it.

## Components

### 1. Orchestrator (`furrow/core/orchestrator.py`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop** (one cycle):
  1. `cycles += 1`, emit `cycle_start` with the current goal, a short `focus` and
     the number of tasks tracked so far
  2. Plan - `PlannerAgent.plan(current_goal)` returns a `Plan`; its tasks are
     merged into `self.tasks` keyed by id (re-planning a task replaces the
     previous version) and `tasks_planned_last_cycle` is recorded
  3. Execute - `_execute` runs the plan through the dependency-aware scheduler
  4. Test - `TesterAgent.run(original_goal, plan.tasks)` returns a `TestResult`;
     pass resets `consecutive_failures` to 0, fail increments it and rewrites
     `current_goal`
  5. Emit `cycle_end`, then check the halt conditions
- **Halt conditions** (checked in this order, first match wins; each emits a
  `halted` event with its reason and breaks the loop):
  1. `goal complete: the planner has no work left to plan` - `_is_done()` is true
     when nothing has been planned at all, or when the planner returned zero tasks
     this cycle. Note this is *not* "all tracked tasks completed".
  2. `failure limit reached: N consecutive cycles with failing tests` - checked
     only when `max_consecutive_failures > 0`; the counter is zeroed by any
     passing cycle, so alternating pass/fail never trips it.
  3. `cycle budget reached: N/M cycles` - checked only when `max_cycles > 0`.
     `0` means unlimited.

  The loop deliberately does **not** halt just because one cycle succeeded, and a
  failed worker does not end the run: that task is marked `failed` and the cycle
  finishes. Three conditions, or `Ctrl-C` from the CLI.
- **Goals:** `original_goal` is set once in `__init__` and never overwritten.
  `current_goal` is what the agents are given, and on a failing test it becomes
  `"Original goal: <original>\n\nFix these failing tests:\n<failures>"`. The tester
  is always handed `original_goal`. Without that split, a few failing cycles would
  leave the agents optimising a repair note instead of the user's actual target.
- **Scheduling:** `_execute` is a ready-queue loop, not a fire-all-and-wait.
  Dependencies are resolved against the current plan only; a task launches as soon
  as every dependency has state `completed`, up to `cap = max(1,
  max_parallel_tasks)`, then the loop awaits the first completion and
  re-evaluates. `asyncio.wait(..., FIRST_COMPLETED)` is used rather than
  `as_completed` so one worker's exception is caught and recorded instead of
  aborting the cycle. A plan with no dependencies degenerates to "keep the cap
  full", i.e. the previous behaviour: ordering must not cost parallelism.
  - The `asyncio.Semaphore(max(1, max_parallel_tasks))` created in `run()` is the
    hard ceiling on real concurrency; the cap above only decides how much work is
    queued, so a dependent task is not even created before its turn is legitimate.
  - The graph is load-bearing because `WorkerAgent` writes *whole* files: two
    workers on one file would perform an unsynchronised read-modify-write and
    silently lose an edit. The planner is asked for `dependencies` precisely so
    that never happens.
  - Malformed plans cost a cycle, not a run: unknown dependency ids become a
    `dependency_warning` with reason `unknown_dependency`, and cycles found by
    Kahn's algorithm (`_orderable`) become a `dependency_warning` with reason
    `cycle`. Tasks that never run are marked `skipped` (`SKIPPED`) - neither
    `completed` nor `failed`, because nothing was attempted - with an
    `unmet dependency: ...` result, including tasks still pending at the end of
    the cycle because an upstream task failed.
- **Events:** `on_event` is an optional synchronous `Callable[[dict], None]`. Every
  event is stamped with `type` and `cycle`, and exceptions raised by the hook are
  swallowed so an observer cannot abort a run. Types:
  - `cycle_start` - `goal`, `focus`, `tasks_tracked`
  - `plan` - `task_ids`, `count`, `rationale`, `tasks_tracked`
  - `task_completed` - `id`, `status` (`completed` / `failed` / `skipped`),
    `result` (truncated to 200 characters)
  - `test_result` - `passed`, `summary`, `failures`, `consecutive_failures`
  - `cycle_end` - `goal`, `tasks_completed`, `tasks_failed`
  - `halted` - `reason`, `cycles`, `consecutive_failures`
  - `dependency_warning` - `reason`, `message`, plus `refs` (unknown ids) or
    `tasks` (cycle members)

  `furrow/web/server.py` is built on this hook: it encodes each event into one
  websocket frame, and adds two terminal frames of its own, `done` and `error`.

### 2. Planner (`furrow/agents/planner.py`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** `Plan` with `tasks[]` and a `rationale`; each `TaskModel` has `id`,
  `description`, `files`, `dependencies`, plus mutable `status` and `result`. The
  prompt asks for 1-5 tasks, each completable in 1-3 tool call rounds.
- **Failure mode:** a reply that will not parse raises `ValueError` quoting the
  response. Unlike the tester, the planner has no heuristic fallback.

### 3. Worker (`furrow/agents/worker.py`)
- **Type:** Subagent
- **Role:** Executes a single assigned task - and it is the **only component in
  Furrow that edits files**. The previous version of this doc implied workers
  reported without doing anything; that is no longer true.
- **Input:** `TaskModel.description` and `TaskModel.files`.
- **File context:** reads the assigned files through `LLMClient.read_file` and
  renders them as a delimited block, bounded by `MAX_CONTEXT_FILES = 20` files and
  `MAX_CONTEXT_CHARS = 20_000` characters. Paths past the file cap and files that
  could not be read are named as notes inside the block (notes come first, so
  they survive truncation) - that tells the model a file is about to be created
  rather than that it is empty.
- **Output contract:** exactly one JSON object, no fences, no prose, parsed with
  `extract_json` (which tolerates fences and surrounding prose anyway):

  ```json
  {"summary": "one line describing what changed",
   "edits": [{"path": "src/auth.py", "content": "<complete new file text>"}]}
  ```

  `content` is the *complete* new contents of the file, never a diff or a
  fragment. A missing `summary` falls back to a default string; a non-list
  `edits`, a non-object edit, or a non-string `path`/`content` raises `ValueError`
  quoting the raw reply, and the orchestrator marks the task `failed`.
- **Writes:** each edit goes verbatim through `LLMClient.write_file` (which
  creates parent directories), and the paths actually written are recorded in
  `WorkerAgent.written_files`. The returned string is `"<summary>\nWrote: <paths>"`.
- **No tool-calling loop:** the model sees the files once, replies with whole file
  bodies, and those bodies are applied as-is. That keeps the write surface
  auditable, and it is the reason two workers must never touch the same file
  concurrently.
- **`dry_run`:** `WorkerAgent(task=..., dry_run=True)` parses the envelope and
  reports the intended edits without writing anything; `written_files` stays
  empty. Not exposed by the CLI.
- **Constraint:** 1-3 tool call rounds per task - in practice one completion.

### 4. Tester (`furrow/agents/tester.py`)
- **Type:** Subagent
- **Role:** Runs the project's test suite and turns the output into a pass/fail
  verdict. It does **not** run lint or type checks.
- **Runner detection:** `TEST_COMMANDS` is tried in order and the first one that
  exists wins: `pytest -q`, `python -m pytest -q`, `npm test -- --silent`,
  `pnpm test -- --silent`, `yarn test --silent`, `cargo test -q`,
  `go test ./...`. Only `FileNotFoundError`, `NotADirectoryError` and
  `PermissionError` mean "not installed here" and move on to the next candidate;
  any other error is a real problem and is not swallowed. If nothing is found the
  output is `No test runner found.`
- **cwd:** the subprocess runs in `settings.workspace` (defaulting to
  `Path.cwd()`), so the runner sees the project's own tests.
- **Return codes:** the output is prefixed with `$ <command>` and
  `exit code: N (passed|FAILED)`, and that hint is shown both to the model and at
  the top of the resulting `TestResult.summary`.
- **Timeout reaping:** `TEST_TIMEOUT_SECONDS = 120`. On timeout the child is
  killed and then awaited, so it does not linger as a zombie, and the output
  records `timed out after 120s; process killed`.
- **Verdict:** the model is asked for `{"passed", "summary", "failures"}` and the
  reply is parsed with `extract_json`. An unparseable or schema-violating reply
  falls back to a crude `"passed" in response.lower()` heuristic (with an empty
  `failures` list) rather than failing the cycle; an exception while running the
  suite becomes a failed `TestResult` with the exception message as the summary.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from
  `$ARGUMENTS`, asking the user for one if none was given.

## Files
```
.kilo/
  FURROW.md            # this document
  agent/
    furrow.md          # Orchestrator
    furrow-planner.md  # Planner subagent
    furrow-worker.md   # Worker subagent
    furrow-tester.md   # Tester subagent
  command/
    furrow.md          # /furrow slash command
furrow/
  __init__.py          # exports LLMClient, Settings
  config.py            # Settings, Provider, Plan, TaskModel, TestResult
  llm.py               # LLMClient (providers, retries, file IO), extract_json
  agents/
    prompts.py         # PLANNER_PROMPT, WORKER_PROMPT, TESTER_PROMPT
    planner.py
    worker.py
    tester.py
  core/
    orchestrator.py    # Orchestrator, _orderable, SKIPPED
  cli/
    main.py            # `furrow` console script: `start` and `web`
  web/
    server.py          # FastAPI app: GET /, websocket /ws, run()
kilo.json              # Project config (default agent = furrow, open permissions)
pyproject.toml
tests/                 # pytest suite; asyncio_mode = "auto"
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- `/furrow <goal>` starts the loop; `furrow start "<goal>"` does the same from the
  shell, and `furrow web` serves the loop behind a browser UI.
- The run ends on one of the orchestrator's three halt conditions, or when the
  user stops it (`Ctrl-C` in the CLI).
- Mid-loop user input is incorporated into the next cycle's plan.

## Future Work
- Structured logging. `structlog` is declared in `pyproject.toml` and
  `Settings.log_level` exists, but neither is read anywhere: the orchestrator
  prints through a `rich` `Console` and the CLI through `rich` too.
- Make `--model` and the web UI's model field actually take effect. They set
  `Settings.model`, but the planner, worker and tester each pass an explicit
  per-role model, so the override currently has no effect on agent calls.
- Declare `request_timeout` and `max_tokens` as real `Settings` fields. `llm.py`
  reads both with `getattr` and falls back to hard-coded 120s / 4096, so they are
  not configurable today.
- Serialise writes per file. The dependency graph is currently the only thing
  stopping two workers from clobbering the same whole-file rewrite.
- A native desktop shell. The CLI and the web UI cover the non-TUI cases today.
