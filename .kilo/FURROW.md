# Furrow Agent Architecture

> **Two layers, one idea.** This file documents the **Kilo subagent layer** —
> the markdown agents and slash command under `.kilo/`, which a *human* drives
> inside a Kilo session. It is **not** a description of the `furrow` Python
> package, which is a standalone agent driven from a terminal or a browser and
> needs no Kilo session at all. Both layers run the same plan → parallel execute
> → test loop and are documented separately below; don't assume a fact about one
> holds for the other.

## Overview

Furrow is an autonomous coding agent that runs a development loop for
web/CLI/desktop projects: plan the goal into independent tasks, execute them in
parallel, verify, feed the failures back into the next plan, repeat.

- **Kilo layer** — `.kilo/agent/*.md` + `.kilo/command/furrow.md`. Started with
  `/furrow <goal>`, orchestrated by the `furrow` primary agent spawning
  `furrow-planner` / `furrow-worker` / `furrow-tester` subagents via the Task
  tool. State lives in the agent's context and is compacted when it grows.
- **Python layer** — the `furrow/` package. Started with `furrow start "<goal>"`
  or `furrow web`, orchestrated by `furrow.core.orchestrator.Orchestrator`
  driving the planner/worker/tester agents against an LLM API, with progress
  persisted to JSON on disk. See [The Python package](#the-python-package).

## Kilo layer components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Report and repeat

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Fixes failures. Returns pass/fail JSON.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

## Kilo layer files

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
```

## Kilo layer permissions

Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Kilo layer usage

- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## The Python package

The installable `furrow` package is a separate, self-contained implementation of
the same loop. It does not use the `.kilo/` agents; those describe Kilo
subagents, while the package drives LLM API calls itself.

### Cycle

Each cycle is plan → execute → test:

1. `PlannerAgent.plan(goal, context)` returns a `Plan` of `TaskModel`s. The
   previous cycle's failures arrive as `context`, so a plan reacts to what broke.
2. Tasks run in dependency waves under an `asyncio.Semaphore` sized by
   `max_parallel_tasks`. Bad dependencies mark tasks `blocked` and the blocking
   propagates transitively — as it does for a dependency that ended `failed` or
   `skipped`, since work whose prerequisite died can never run; a stop request
   or a wave that changed nothing marks never-dispatched tasks `skipped`.
   `max_task_attempts` escalates a failing task
   to `skipped`, but because every cycle re-plans (fresh tasks, `attempts = 0`)
   that budget only bites when it is set to `1` — retries come from the next
   plan, not from it.
3. `TesterAgent.run(goal, tasks)` runs the project's suite (auto-detected, or
   `FURROW_TEST_COMMAND`), appends `ruff check .` / `mypy furrow` output when
   available, and asks the model for a `TestResult` verdict. A cycle where
   nothing could be verified is reported as passed, not as a failure.

The loop exits when every task is `completed` **and** the verdict passed, when
`max_cycles` is reached, when `max_consecutive_failures` cycles in a row fail
identically, when the planner has failed to produce a plan on
`max_consecutive_failures` cycles in a row (reported as an `error` event, never
as success — an empty plan the planner returned is a finished goal, an empty plan
its failure produced is not), or on `stop()` / `CancelledError`. The goal itself
is never rewritten on failure — only the focus context is.

### `on_event` contract

`Orchestrator(goal, client, on_event, settings)` accepts
`on_event: Callable[[str, dict[str, Any]], None]`. It is invoked
synchronously from inside the loop and is best-effort: an exception raised by
the callback is swallowed and never breaks a run. Events:

| Event | Payload |
| --- | --- |
| `cycle_start` | `{"cycle": int}` |
| `plan` | `{"rationale", "task_count", "tasks": [...]}` |
| `task_start` | `{"id", "description"}` |
| `task_done` | `{"id", "status", "result", "error"}` |
| `test_result` | `{"passed", "summary", "failures"}` |
| `cycle_end` | `{"cycle", "passed", "completed": [ids], "failed": [ids]}` |
| `done` | `{"reason", "cycles"}` — last event of a clean exit |
| `error` | `{"message", "where"}` — a recovered planner/tester failure |

### Shipped capabilities (Python package)

- **Persistent run state.** Every cycle's goal, task list (status, result,
  error, attempt count) and per-cycle history (most recent 50 cycles) is written
  atomically to `<workspace>/.furrow/state.json` and deleted once the goal is
  achieved. Reads and writes are best-effort, so a missing or corrupt file never
  crashes a run. Override the location with `FURROW_STATE_FILE` or `--state-file`.
- **Web UI for monitoring.** `furrow web` serves a page that streams the
  orchestrator's events over a websocket, so a long run shows live progress and
  can be stopped from the browser.

## Future Work

- CLI and desktop wrappers for the **Kilo layer**, so the subagent loop can be
  driven without a Kilo session. (The Python package already has a CLI; what is
  missing is a wrapper around the markdown agents themselves.)
- Richer monitoring views for the web UI — task trees, per-task diffs, history
  browsing across runs. Today it streams the raw event log.
- Reusing a persisted run's state to resume a goal in a later session; the
  state file is currently written and cleared but not read back at startup.
