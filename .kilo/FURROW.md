# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs a plan/execute/test loop for web/CLI/desktop projects.

There are two implementations of that loop, and they are worth keeping apart:

- **Python package (`furrow/`)** — the `pip install furrow` distribution. The
  orchestrator drives `PlannerAgent`, `WorkerAgent`, and `TesterAgent` in-process
  via asyncio and is what `furrow start` and `furrow web` run. See `README.md`.
- **Kilo agents (`.kilo/`)** — the agent/command definitions below, used by
  `/furrow` inside the TUI. The orchestrator here delegates with the Task tool
  rather than asyncio.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Report and repeat

In the Python orchestrator (`furrow/core/orchestrator.py`) each step is: plan
with `PlannerAgent`, run one `WorkerAgent` per new task under an
`asyncio.Semaphore` capped by `settings.max_parallel_tasks`, then run the test
suite. Tasks are de-duplicated by `id` across cycles, and a task that already
settled as completed or skipped is never dragged back to pending. The original
goal is never mutated — previous test failures are appended to the planner input
as context instead.

The loop halts on the first of: `stop()` was called, the planner returned an
empty task list (the prompt defines that as "goal complete"), every tracked task
settled with none failed and no work outstanding, `max_cycles` was reached
(`0` = unlimited), or the planner/tester raised — which is recorded as
`stopped_reason="error"` rather than propagated.

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.

`PLANNER_PROMPT` (`furrow/agents/prompts.py`) also constrains `status` to one of
`pending`, `running`, `completed`, `failed`, `skipped` and states that returning
an empty `tasks` array is the correct answer when the goal is complete.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Handles a single assigned task: minimal, targeted changes, no scope creep.
- **Constraint:** 1-3 tool call rounds per task.

The `.kilo/agent/furrow-worker.md` subagent has tool access and does edit files.
The Python `WorkerAgent` (`furrow/agents/worker.py`) does not: `WORKER_PROMPT`
opens with "You have NO tool access and cannot edit files directly" and asks for
a concise implementation summary of what to change and in which files, for "a
follow-up actor" to execute. `WorkerAgent.run()` returns that text, which the
orchestrator stores as `TaskModel.result`. So in `furrow start` / `furrow web` the
workers produce plans, not diffs.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs the project's test suite, judges the result, and reports
  pass/fail JSON.
- **No fixing:** `TesterAgent` does not edit code. It executes
  `settings.test_command` — or, when that is unset, the first available of
  `pytest -q`, `python -m pytest -q`, `npm test`, `pnpm test`, `yarn test`,
  `cargo test -q`, `go test ./...` — inside `settings.workspace`. An explicit
  `test_command` is bounded by `settings.test_timeout`; the auto-detected path
  uses a hardcoded 120s.
- **Exit code is authoritative:** `result.passed = exit_code == 0` is assigned
  after parsing, so the model cannot talk the suite into passing. `TESTER_PROMPT`
  says the same and forbids contradicting the code. A missing runner (exit code
  absent) counts as inconclusive, not a pass.
- **The model only writes prose:** the human-readable `summary` and the
  `failures[]` entries, which the planner sees on the next cycle.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

Defined at `.kilo/command/furrow.md` (`agent: furrow`, `subtask: false`); it asks
the user for a goal if `$ARGUMENTS` is empty.

## Files
```
.kilo/
  agent/
    furrow.md          # Orchestrator
    furrow-planner.md  # Planner subagent
    furrow-worker.md   # Worker subagent
    furrow-tester.md   # Tester subagent
  command/
    furrow.md          # /furrow slash command
kilo.json              # Project config (default agent = furrow, open permissions)
furrow/                # Python package: cli/, core/orchestrator.py, agents/, web/, llm.py, config.py
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

Outside the TUI: `furrow start [GOAL]` and `furrow web`, both documented in
`README.md`.

## Known limitations
- **The Python worker cannot edit files.** See section 3 — `WorkerAgent` is a
  summariser, so nothing in `furrow start` / `furrow web` currently applies the
  work it plans.
- **No state persists across sessions.** Tasks, cycle count, failures, and test
  results live in a `RunResult` in memory and are discarded when `run()` returns.
- **`structlog` is declared but unused.** It is a dependency in `pyproject.toml`
  and nothing imports it; the code uses stdlib `logging` via
  `logging.getLogger(__name__)`.
- **Settings are resolved once, at import.** `furrow/config.py` builds
  `settings = Settings()` at module scope, so changing `FURROW_*` variables after
  import has no effect. CLI model/provider overrides go into a *new* `Settings`
  handed to `LLMClient`, but `--max-parallel-tasks` has to be written onto the
  global singleton because the orchestrator reads parallelism from there.

## Future Work
- Give the worker agent tool access so a cycle applies its own changes, closing
  the gap described in section 3.
- Persistent state file for goal/task tracking across sessions.
- Desktop wrapper for non-TUI interaction; the CLI wrapper already exists.
- A richer web UI than the single-page console `furrow web` ships today: it
  streams progress over a websocket at `/ws`, but has no multi-run view, history,
  or task-level detail.
