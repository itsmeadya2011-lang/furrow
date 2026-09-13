# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects.

## Components

### 1. Orchestrator (`furrow/core/orchestrator.py`)
- **Type:** Primary agent
- **Role:** Runs the infinite loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Report and repeat
- **Features:**
  - Persistent plan and task state across cycles
  - `max_cycles` enforcement (safety limit)
  - `request_stop()` for graceful shutdown (used by web UI)
  - Structured logging via `structlog`
  - Task result tracking via `task_results` property

### 2. Planner (`furrow/agents/planner.py`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.

### 3. Worker (`furrow/agents/worker.py`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.

### 4. Tester (`furrow/agents/tester.py`)
- **Type:** Subagent
- **Role:** Runs tests, linting, and type checks. Fixes failures. Returns pass/fail JSON.
- **Features:** Respects `settings.workspace` for test execution directory.

### 5. LLM Client (`furrow/llm.py`)
- **Type:** Client
- **Role:** Abstracts LLM provider calls with retry logic.
- **Features:**
  - Supports Anthropic, OpenAI, and Ollama providers
  - Retry with exponential backoff via `tenacity`
  - File I/O via `aiofiles`

### 6. Web UI (`furrow/web/server.py`)
- **Type:** FastAPI application
- **Role:** Web interface for monitoring and controlling Furrow.
- **Features:**
  - WebSocket-based real-time output streaming
  - Stop button for graceful orchestrator shutdown
  - Active orchestrator tracking

### 7. CLI (`furrow/cli/main.py`)
- **Type:** Click CLI
- **Role:** Entry point for running Furrow from the terminal.
- **Commands:** `furrow start <goal>`, `furrow web`

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
AGENTS.md              # This file
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## Future Work
- CLI and desktop wrappers for non-TUI interaction.
- Persistent state file for goal/task tracking across sessions.
- Web UI for monitoring parallel agent activity.
