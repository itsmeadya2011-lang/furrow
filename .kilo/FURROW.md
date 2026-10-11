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

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.
- **Tools:** `read_file`, `write_file`, `list_files`, `run_command`, `complete`

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

## Configuration
Environment variables (prefix `FURROW_`):
- `PROVIDER`: `anthropic`, `openai`, `ollama`
- `MODEL`, `PLANNER_MODEL`, `WORKER_MODEL`, `TESTER_MODEL`
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OLLAMA_BASE_URL`
- `MAX_PARALLEL_TASKS` (default: 5)
- `MAX_CYCLES` (default: 0 = unlimited)
- `WORKSPACE` (default: cwd)
- `LOG_LEVEL` (default: INFO)
- `TEST_COMMAND` (custom test runner)

## Implemented Improvements
- ✅ Persistent state file (`.furrow_state.json`) for goal/task tracking across sessions
- ✅ Task dependency handling (tasks wait for dependencies to complete)
- ✅ Ollama provider support
- ✅ Retry logic with exponential backoff for LLM calls (via tenacity)
- ✅ Structured logging with structlog
- ✅ WebSocket streaming in web UI
- ✅ Configurable test command
- ✅ Max cycles and max parallel tasks enforcement
- ✅ Worker agents with actual file operation tools

## Future Work
- CLI and desktop wrappers for non-TUI interaction.
- Web UI for monitoring parallel agent activity with real-time task visualization.
- Test failure auto-fix via worker agents.
