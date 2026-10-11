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
- **State Management:** Persistent state file (`.furrow_state.json`) tracks goals, tasks, history, and progress across sessions. Supports `--resume` flag.

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.
- **Context Awareness:** Receives project file structure, task history, and current task statuses.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task using tool calls (read_file, write_file, list_files, run_command).
- **Constraint:** 1-3 tool call rounds per task.
- **Implementation:** Actually modifies files and runs commands, not just text generation.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Attempts to fix failures by creating fix tasks.
- **Auto-fix:** When tests fail, creates targeted fix tasks and re-runs tests.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

### 6. Web UI (`furrow web`)
- **Type:** FastAPI + WebSocket server
- **Role:** Real-time monitoring of parallel agent activity with task progress, cycle updates, and test results.

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
- Run `furrow start "goal"` from CLI.
- Run `furrow web` to start web UI at http://localhost:8000.
- Use `--resume` flag to continue from saved state.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## Configuration
Environment variables (prefix `FURROW_`):
- `FURROW_PROVIDER`: anthropic | openai | ollama
- `FURROW_MODEL`: Model name (default: claude-sonnet-4-20250514)
- `FURROW_ANTHROPIC_API_KEY` / `FURROW_OPENAI_API_KEY`: API keys
- `FURROW_OLLAMA_BASE_URL`: Ollama endpoint (default: http://localhost:11434)
- `FURROW_MAX_PARALLEL_TASKS`: Max concurrent workers (default: 5)
- `FURROW_MAX_CYCLES`: Max cycles before stopping (0 = unlimited)
- `FURROW_WORKSPACE`: Working directory
- `FURROW_LOG_LEVEL`: DEBUG | INFO | WARNING | ERROR
- `FURROW_STATE_FILE`: State file path (default: .furrow_state.json)

## Completed Improvements
- ✅ Persistent state management with `.furrow_state.json`
- ✅ Proper task tracking and completion logic
- ✅ Worker executes actual code changes via tool calls
- ✅ Planner with project context awareness
- ✅ Max cycles enforcement and better stopping conditions
- ✅ Tester auto-fixes failing tests
- ✅ Dependency resolution for task ordering
- ✅ Ollama provider support
- ✅ Structured logging with structlog
- ✅ Web UI with real-time task progress
