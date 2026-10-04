# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects.

The loop is unbounded by default. `FURROW_MAX_CYCLES` (`0` = unbounded) is the runaway guard, and
`FURROW_MAX_PARALLEL_TASKS` (default `5`) caps concurrent workers per cycle.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the infinite loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Report and repeat
- **Halts when:** the plan is non-empty, every task completed, and the tests passed. An empty plan
  or a failing test means another cycle.

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.
- **Note:** the Python `WorkerAgent` returns a text summary and does not touch the filesystem. Edits
  come from these `.kilo/` definitions, which have real tool access.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Fixes failures. Returns pass/fail JSON.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

### 6. Event seam (`furrow/core/orchestrator.py`)
- **Type:** Instrumentation point
- **Role:** Every orchestrator message goes through `_emit`, which prints it and forwards a JSON
  payload to an optional event sink. Terminal output and the sink cannot drift apart.
- **Used by:** `furrow web`, which streams the sink over a `/ws` websocket to a browser page showing
  the cycle counter, with a Stop button. A client disconnect cancels the run instead of leaking it.

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
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete, `FURROW_MAX_CYCLES` cycles have run, or the user stops
  the session.
- Mid-loop user input is incorporated into the next cycle's plan.
- Non-TUI front-ends already exist: `furrow start "<goal>"` and `furrow web` (`--cycles`,
  `--max-parallel`, plus `--model` on `start` and `--host` / `--port` on `web`).

## Future Work
- Persistent state file for goal/task tracking across sessions.
- Desktop wrapper on top of the existing CLI and web front-ends.
