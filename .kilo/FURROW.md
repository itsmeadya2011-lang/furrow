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
furrow/                # Python package (CLI + programmatic orchestrator)
  cli/main.py          # CLI entry point
  core/orchestrator.py # Python orchestrator (plan → execute → test → repeat)
  core/state.py        # Persistent state save/load
  agents/              # Planner, Worker, Tester Python agents
  llm.py               # Async LLM client (Anthropic/OpenAI/Ollama + retry)
  web/server.py        # FastAPI + WebSocket server
  config.py            # Pydantic settings
```

## Configuration
Settings are configurable via environment variables (prefixed `FURROW_`) or a `.env` file:
- `provider` — `anthropic` | `openai` | `ollama` (default: `anthropic`)
- `model`, `planner_model`, `worker_model`, `tester_model`
- `max_cycles` — max development cycles (0 = unlimited)
- `max_parallel_tasks` — concurrent worker tasks (default: 5)
- `ollama_base_url` — Ollama API URL
- `state_file` — path to the JSON state file (default: `.furrow-state.json`)
- `log_level` — logging verbosity

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## Future Work
- Desktop wrapper for non-TUI interaction.
- Web UI for monitoring parallel agent activity (web server is functional; richer dashboard planned).
