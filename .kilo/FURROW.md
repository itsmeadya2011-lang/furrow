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

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Fixes failures. Returns pass/fail JSON.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

### 6. CLI (`furrow` command)
- **Type:** Python CLI
- **Commands:** `start`, `resume`, `web`
- **Features:** State persistence, cycle limits, structured logging

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
furrow/
  cli/main.py          # CLI entry point
  core/orchestrator.py # Orchestrator implementation
  agents/              # Planner, Worker, Tester agents
  llm.py               # LLM client with retry logic
  config.py            # Configuration models
  web/server.py        # Web UI with WebSocket
kilo.json              # Project config (default agent = furrow, open permissions)
pyproject.toml         # Python package config
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop in Kilo.
- Run `furrow start <goal>` to start the loop from CLI.
- Run `furrow resume` to resume from a saved state file.
- Run `furrow web` to start the web UI.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.
- State is persisted to `.furrow_state.json` after each cycle.

## Configuration
Environment variables (prefix `FURROW_`):
- `FURROW_PROVIDER`: `anthropic` | `openai` | `ollama`
- `FURROW_MODEL`: Default model
- `FURROW_PLANNER_MODEL`: Model for planning
- `FURROW_WORKER_MODEL`: Model for workers
- `FURROW_TESTER_MODEL`: Model for testing
- `FURROW_ANTHROPIC_API_KEY`: Anthropic API key
- `FURROW_OPENAI_API_KEY`: OpenAI API key
- `FURROW_OLLAMA_BASE_URL`: Ollama base URL
- `FURROW_MAX_PARALLEL_TASKS`: Max parallel tasks (default 5)
- `FURROW_MAX_CYCLES`: Max cycles (0 = unlimited)
- `FURROW_WORKSPACE`: Workspace path
- `FURROW_LOG_LEVEL`: Log level

## Future Work
- CLI and desktop wrappers for non-TUI interaction.
- Persistent state file for goal/task tracking across sessions. ✅ Done
- Web UI for monitoring parallel agent activity. ✅ Done (basic)
- Integration with Kilo subagents for planner/worker/tester. ✅ Done
- Structured logging with structlog. ✅ Done
- Retry logic for LLM calls. ✅ Done