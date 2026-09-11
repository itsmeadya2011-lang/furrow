# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects. Each cycle follows a structured pipeline: **Plan → Execute (parallel) → Test → Retry/Recover → Persist State → Report → Repeat**.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent
- **Role:** Runs the development loop. Receives a goal, delegates to the planner subagent, dispatches parallel workers, invokes the tester, manages retries and state persistence, and repeats until the goal is complete or the user stops the session.
- **Loop:**
  1. **Receive / Identify** — Load the current goal from user input or `.kilo/furrow-state.json`. Load project context (test/lint commands).
  2. **Plan** — Invoke the `furrow-planner` subagent. Parse its JSON output (`tasks[]`, `rationale`, `multi_cycle`).
  3. **Execute** — Sort tasks by priority (`high` → `medium` → `low`). Spawn `furrow-worker` subagents in parallel for independent tasks. Resolve dependencies incrementally. Parse worker JSON output.
  4. **Test** — Invoke the `furrow-tester` subagent. Parse `passed`, `summary`, `failures[]`.
  5. **Retry / Recover** — If a worker or test fails, re-spawn workers with fix instructions. Max 3 retries per task (shared counter).
  6. **Persist State** — Write updated state to `.kilo/furrow-state.json`.
  7. **Report** — Summarize what was built, tested, and what remains.
  8. **Repeat** — If `multi_cycle: true` or tasks remain, begin the next cycle.

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable tasks with priority, effort, and acceptance criteria.
- **Output:** JSON with `tasks[]` (each having `id`, `description`, `files`, `dependencies`, `priority`, `effort`, `acceptance_criteria`), plus top-level `rationale` (string) and `multi_cycle` (boolean).

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Output:** JSON with `task_id`, `completed` (boolean), `changes` (string[]), `files_modified` (string[]), `issues` (string[]).
- **Constraint:** 1–3 tool call rounds per task.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Discovers and runs the project's test suites, linters, and type checkers. Fixes failures when possible.
- **Output:** JSON with `passed` (boolean), `summary` (string), `failures[]` (array of objects with `task_id`, `description`, `fix_suggestion`).

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Resolves the goal in three tiers:
  1. If `$ARGUMENTS` is provided → set as the active goal.
  2. If no argument → load the goal from `.kilo/furrow-state.json` (resume previous session).
  3. If neither → ask the user for a goal.
  Then hands off to the `furrow` orchestrator.

## JSON Contracts

| Component | Output Fields |
|-----------|--------------|
| **Planner** | `tasks[]`: `id`, `description`, `files`, `dependencies`, `priority`, `effort`, `acceptance_criteria` · `rationale` · `multi_cycle` |
| **Worker** | `task_id` · `completed` · `changes[]` · `files_modified[]` · `issues[]` |
| **Tester** | `passed` · `summary` · `failures[]`: `task_id`, `description`, `fix_suggestion` |
| **State** | `goal`, `current_cycle`, `multi_cycle`, `tasks[]`, `cycle_history[]` |

## State File Format

The state file `.kilo/furrow-state.json` is read at the start of each cycle and written at the end. It is the source of truth for goal, task status, and cycle history across sessions.

```json
{
  "goal": "the current high-level objective",
  "current_cycle": 1,
  "multi_cycle": false,
  "tasks": [
    {
      "id": "1",
      "description": "...",
      "priority": "high",
      "effort": 1,
      "acceptance_criteria": "...",
      "status": "pending | in_progress | completed | failed",
      "retries_used": 0,
      "files": ["..."],
      "dependencies": [],
      "worker_result": {
        "task_id": "1",
        "completed": true,
        "changes": [
          { "path": "...", "content": "..." }
        ],
        "files_modified": ["..."],
        "issues": []
      }
    }
  ],
  "cycle_history": [
    {
      "cycle": 1,
      "plan_rationale": "...",
      "multi_cycle": false,
      "tasks_completed": ["1"],
      "tasks_failed": [],
      "worker_results": [],
      "test_result": {
        "passed": true,
        "summary": "...",
        "failures": []
      },
      "passed": true
    }
  ]
}
```

### State field notes
- `tasks[].status`: `pending` (not yet dispatched), `in_progress` (worker running or retrying), `completed` (worker returned `completed: true`), `failed` (retries exhausted or blocked).
- `tasks[].retries_used`: increments on each worker re-spawn for that task; caps at 3.
- `cycle_history[]`: one entry per completed cycle, preserving rationale, worker results, test results, and which tasks succeeded/failed that cycle.

## Retry Policy
- Each task has a retry budget of 3.
- Retries are shared across worker execution failures and test failures for the same task.
- If a worker returns `completed: false`, the orchestrator re-spawns a `furrow-worker` with fix instructions derived from the `issues` field.
- If the tester returns `passed: false`, the orchestrator parses `failures[].task_id` to identify which tasks need fixing and re-spawns those workers.
- After the budget is exhausted, the task is marked `failed` in state and the loop continues with remaining tasks.

## Multi-Cycle Handling
- When the planner returns `multi_cycle: true`, the goal requires more than one planning/execution/test cycle.
- The orchestrator continues looping, re-invoking the planner each cycle to plan the next slice, until `multi_cycle` is `false` and the goal is satisfied.
- The `cycle_history` in the state file tracks progress across cycles.
- If `multi_cycle: false` and all tasks completed with tests passing, the goal is marked complete in state and the loop exits.

## Workflow
```
User Goal → /furrow command → furrow orchestrator
    ↓
   [Load state from .kilo/furrow-state.json]
    ↓
┌─────────────────────────────────────────────┐
│ 1. PLAN:  furrow-planner → tasks[] JSON     │
│ 2. EXECUTE: furrow-worker (parallel)        │
│ 3. TEST:  furrow-tester → passed/failures   │
│ 4. RETRY: re-spawn failed workers (≤3x)     │
│ 5. PERSIST: write .kilo/furrow-state.json   │
│ 6. REPORT + Repeat if multi_cycle or work   │
└─────────────────────────────────────────────┘
```

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
  FURROW.md             # This document
kilo.json               # Project config (default agent = furrow, open permissions)
.kilo/furrow-state.json # State file (created at runtime, gitignored)
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop with a new goal.
- Run `/furrow` with no arguments to resume the previous goal from state.
- The agent runs until the goal is complete (`multi_cycle: false`, all tasks done, tests passing) or the user stops the session (Ctrl+C / closing the session).
- Mid-loop user input is incorporated into the next cycle's plan.

## Future Work
- CLI and desktop wrappers for non-TUI interaction.
- Web UI for monitoring parallel agent activity.
