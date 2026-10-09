# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects.

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
- **Behavior:**
  - Tracks task state across the cycle
  - Emits structured events (start, cycle, plan, task_start, task_result, tests, cycle_end, done, error)
  - Halts when tests pass / no tasks planned / max_cycles reached / stop requested
  - Preserves the original goal when retrying failing tests
  - Persists state after every cycle to `.furrow/state.json`

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.
- **Details:**
  - Robust JSON extraction (markdown fences tolerated)
  - Task ids normalized to strings
  - Plan capped at `max_parallel_tasks`

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Fixes failures. Returns pass/fail JSON.
- **Details:**
  - Project-aware test-runner detection (pytest/npm/pnpm/yarn/cargo/go based on project files)
  - Runs in the configured workspace
  - 120s timeout per runner

### 5. LLMClient
- **Providers:** anthropic, openai, ollama
- **Transient-error retries** with exponential backoff (`max_retries`)

### 6. Command (`/furrow`)
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
- CLI: `furrow start "goal"`, `furrow state`, `furrow reset`, `furrow web`
- Web UI: `furrow web` launches at http://localhost:8000

## Future Work
- Desktop wrapper (GUI) for non-terminal interaction
- Resume a persisted goal across sessions (`furrow resume`)