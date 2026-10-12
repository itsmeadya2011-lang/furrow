# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the development loop across cycles. Receives goals, plans, delegates, tests, and repeats.
- **Cycles:** Tracks tasks across cycles. Enforces `FURROW_MAX_CYCLES` (0 = infinite). The original goal is preserved in a separate field rather than overwritten by fix-goals.
- **Worker edits:** Applies worker-produced edits (`edits: [{path, content}]`) to disk under `FURROW_WORKSPACE`.
- **State:** Persists state to `.furrow/state.json` after every cycle.
- **Events:** Emits typed events (start, cycle_start, plan, task_complete, task_failed, test_result, done, max_cycles_reached) via `orchestrator.on_event` callbacks (sync or async).
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Report and repeat

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Behavior:** Clamps plans to 5 tasks and normalizes task fields.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.
- **Output:** Returns `edits: [{path, content}]` applied to disk by the orchestrator.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Fixes failures. Returns pass/fail JSON reporting the runner and exit code.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

## Files
```
.gitignore              # Root .gitignore (untracks __pycache__, .furrow, etc.)
.kilo/
  agent/
    furrow.md          # Orchestrator
    furrow-planner.md  # Planner subagent
    furrow-worker.md   # Worker subagent
    furrow-tester.md   # Tester subagent
  command/
    furrow.md          # /furrow slash command
.furrow/
  state.json           # Runtime state (gitignored)
kilo.json              # Project config (default agent = furrow, open permissions)
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## Events
The orchestrator emits typed events via `orchestrator.on_event` callbacks,
which may be synchronous or async. Emitted event types:

| Event              | When                                          |
|--------------------|-----------------------------------------------|
| `start`            | Orchestration begins.                         |
| `cycle_start`      | A new cycle begins.                           |
| `plan`             | Planner produced a task plan.                 |
| `task_complete`    | A worker task finished successfully.          |
| `task_failed`      | A worker task failed.                         |
| `test_result`      | Tester returned a pass/fail result.           |
| `done`             | Goal fully complete; loop exits.              |
| `max_cycles_reached` | `FURROW_MAX_CYCLES` exceeded; loop exits.   |

## Future Work
- Per-task timeout enforcement.
- Worker patch/diff application instead of full-file writes.
- Authentication for the web UI.
