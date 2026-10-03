# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs a bounded development loop for web/CLI/desktop projects.

This repository ships **two parallel implementations of the same loop**. They are kept
deliberately separate; know which one you are looking at.

| | Kilo agents (`.kilo/`) | Python runtime (`furrow/`) |
| --- | --- | --- |
| Entry point | `/furrow <goal>` in the Kilo TUI | `furrow start "<goal>"` / `furrow web` |
| Orchestrator | the `furrow` primary agent, in-session | `furrow/core/orchestrator.py` |
| Planner | `furrow-planner` subagent | `furrow/agents/planner.py` |
| Worker | `furrow-worker` subagent | `furrow/agents/worker.py` |
| Tester | `furrow-tester` subagent | `furrow/agents/tester.py` |
| Durability | agent context only | `.furrow/state.json` |
| Guard rails | in the agent prompt | `Settings` + guard code |

Both perform the same cycle — **plan 1-5 parallel tasks → execute them concurrently →
test → feed failures into the next plan → repeat**. Everything below describing a
component names the Kilo view; the equivalent Python module is given in brackets.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent) [`furrow/core/orchestrator.py`]
- **Role:** Runs the loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents [`furrow/agents/worker.py`]
  4. Test via `furrow-tester` subagent [`furrow/agents/tester.py`]
  5. Report and repeat
- **Termination conditions** (the Python runtime implements all five; the Kilo agent is
  bounded by the user stopping the session):
  1. Goal satisfied — tests passed and every planned task completed.
  2. `max_cycles` budget exhausted (`0` = unbounded).
  3. Stagnation — the same test-failure set repeated `stagnation_limit` times running.
  4. Consecutive-error budget — `max_consecutive_errors` erroring cycles in a row.
  5. Cooperative stop — `Orchestrator.stop()` (web stop button) or Ctrl+C (exit 130).
- **Plumbing** [`furrow/core/events.py`, `furrow/state.py`]: each cycle emits events on an
  in-process `EventBus` (`run_start`, `cycle_start`, `plan`, `task_start`, `task_end`,
  `test_result`, `cycle_end`, `run_end`, `error`, `notice`) and persists a `RunState`
  snapshot after every cycle and at the end of the run. Failures accumulate in
  `fix_instructions` (de-duplicated, capped at 4000 chars) rather than overwriting the
  goal.

### 2. Planner (`furrow-planner`)
- **Type:** Subagent [`furrow/agents/planner.py`]
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.
- **Normalizing:** the Python planner caps the plan at `max_parallel_tasks` tasks,
  rewrites missing/duplicate ids to sequential ones, and drops dependencies that are
  unknown or would create a cycle.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent [`furrow/agents/worker.py`]
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent [`furrow/agents/tester.py`]
- **Role:** Runs tests, lint, type checks. Fixes failures. Returns pass/fail JSON.
- **Deterministic verdict:** in the Python runtime the test command's exit code is
  authoritative (`0` passes, anything else fails, a timeout fails); the model only
  summarizes the output and extracts failure lines.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

## Files
```
furrow/                     # Installable Python runtime (used by `furrow start` / `furrow web`)
  __init__.py               # Lazy re-exports: LLMClient, Settings, Orchestrator, ...
  config.py                 # Settings (FURROW_ env prefix, .env), TaskModel, Plan, TestResult
  state.py                  # RunState + StateStore (atomic JSON persistence)
  llm.py                    # Async anthropic/openai/ollama client, retries, JSON extraction
  agents/
    prompts.py              # PLANNER_PROMPT, WORKER_PROMPT, TESTER_PROMPT
    planner.py              # PlannerAgent
    worker.py               # WorkerAgent
    tester.py               # TesterAgent (test command detection, execution, verdict)
  core/
    events.py               # Event, EventBus
    orchestrator.py         # Orchestrator: the loop and its termination guards
  cli/
    main.py                 # click CLI: start / web / status
  web/
    server.py               # FastAPI app, REST routes, /ws event stream, embedded page
tests/
  test_*.py                 # pytest suite
.kilo/
  FURROW.md                 # This file
  agent/
    furrow.md               # Orchestrator agent definition
    furrow-planner.md       # Planner subagent
    furrow-worker.md        # Worker subagent
    furrow-tester.md        # Tester subagent
  command/
    furrow.md               # /furrow slash command
kilo.json                    # Project config (default agent = furrow, open permissions)
pyproject.toml               # Package metadata, deps, pytest/ruff/mypy config
README.md                    # Install, CLI reference, settings, loop and web UI
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow` (set in `kilo.json`). This is
deliberate — the loop edits and runs code in the workspace without asking.

## Usage
- TUI: run `/furrow <goal>` to start the loop; the agent runs until the goal is complete
  or the user stops the session, and mid-loop user input is folded into the next plan.
- CLI: `furrow start "<goal>"` runs the same loop headlessly; `furrow web` serves the
  monitoring page. See `README.md` for flags, settings and exit codes.

## Future Work
- Resuming a run from a saved state file. `StateStore.load()` can read `.furrow/state.json`
  but nothing reconstructs an `Orchestrator` from it yet.
- A desktop wrapper. Only the CLI and the browser page exist.
- A richer monitoring dashboard. `furrow web` streams a live event log and the final
  state, but has no cross-run history view, metrics or diffing.
