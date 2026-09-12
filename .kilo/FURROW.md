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
  3. Execute via parallel `furrow-worker` subagents (respects max_parallel_tasks)
  4. Test via `furrow-tester` subagent
  5. Report and repeat
- **Lifecycle:** Stores task state across cycles; supports graceful shutdown via `request_stop()`; enforces `max_cycles`

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

### 6. Web UI (`furrow web`)
- **Type:** FastAPI server
- **Role:** Web interface with WebSocket streaming for monitoring parallel agent activity.
- **Endpoints:** `/health` (status), `/ws` (WebSocket for task streaming)

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

## Configuration
- `FURROW_PROVIDER`: LLM provider (anthropic, openai, ollama)
- `FURROW_MODEL`: Model ID for the main LLM
- `FURROW_MAX_CYCLES`: Max cycles (0 = unlimited)
- `FURROW_MAX_PARALLEL_TASKS`: Max concurrent workers (default: 5)
- `FURROW_ANTHROPIC_API_KEY` / `FURROW_OPENAI_API_KEY`
- `FURROW_OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete, max cycles reached, or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## Future Work
- Persistent state file for goal/task tracking across sessions.
- CLI and desktop wrappers for non-TUI interaction.
- Real-time dashboard for monitoring agent activity.
