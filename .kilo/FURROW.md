# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs a plan/execute/test development loop until the goal is complete, the cycle budget is exhausted, or a stop is requested.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the loop for a single goal. Plans, delegates, tests, and repeats.
- **Loop:**
  1. Plan parallel tasks (1-5) via `furrow-planner`
  2. Execute via parallel `furrow-worker` subagents, scheduled in dependency order and bounded by a parallelism cap
  3. Test via `furrow-tester` subagent
  4. Fold failing-check output into the next cycle's plan
  5. Repeat until goal complete, budget exhausted, or stop requested

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, and type checks. Returns pass/fail JSON.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

## Loop mechanics
- Tasks are scheduled in dependency order: the plan is split into topological levels, each running in parallel.
- Parallelism is bounded by `FURROW_MAX_PARALLEL_TASKS` (default 5).
- A cycle budget (`FURROW_MAX_CYCLES`, `0` = unlimited) caps the number of plan/execute/test cycles.
- A stop request (`request_stop` from the CLI or web UI) finishes the current work and halts the loop.
- Progress is observable through an event stream (`run_start`, `plan`, `cycle_start`, `cycle_end`, `task_start`, `task_end`, `test_result`, `stop_requested`, `run_end`) and a serializable `state` snapshot.

## Entry points
- `furrow start "<goal>"` — CLI entrypoint (console script).
- `furrow web` — serves the web UI (`/`, `/api/status`, `/healthz`, `/ws`) with a single-active-run guard.

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
- The agent runs until the goal is complete, the cycle budget is exhausted, or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.
