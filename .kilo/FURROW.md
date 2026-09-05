# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the infinite loop. Receives goals, plans, delegates, tests, and repeats.
- **Features:**
  - **Dependency-aware execution:** tasks with unmet dependencies are deferred; tasks blocked by failed dependencies are marked as failed.
  - **Bounded parallelism:** `max_parallel_tasks` (default 5) via `asyncio.Semaphore`.
  - **Cycle limit:** `max_cycles` (default 0 = infinite) stops the loop after the configured number of cycles.
  - **Persistent state:** goal, cycle count, and task statuses are saved to `.furrow/state.json` after each cycle and loaded on startup.
  - **Structured logging:** `structlog`-based structured logs across all components.
- **Loop:**
  1. Load state from `.furrow/state.json` (if exists)
  2. Receive / Identify goal
  3. Plan parallel tasks (1-5)
  4. Execute via parallel workers, respecting dependencies and concurrency limits
  5. Test via `furrow-tester`
  6. Save state
  7. Report and repeat (until goal complete or max_cycles reached)

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
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## Future Work
- CLI and desktop wrappers for non-TUI interaction.
- Web UI for monitoring parallel agent activity.
- Ollama provider support (config exists, LLMClient not fully wired).
