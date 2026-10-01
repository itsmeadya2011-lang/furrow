# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite parallel development loop for web/CLI/desktop projects.
It exists in two layers: Kilo-native multi-agent definitions (`.kilo/`) and a standalone Python package (`furrow/`).

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent) / Python `Orchestrator` class
- **Role:** Runs the infinite loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents (or `WorkerAgent` in Python)
  4. Test via `furrow-tester` subagent (or `TesterAgent` in Python)
  5. Report and repeat
- **Python Orchestrator features:**
  - `max_parallel_tasks` semaphore bounds concurrent worker execution
  - `max_cycles` limit (0 = infinite) halts the loop after N cycles
  - Event callback (`on_event`) for real-time progress reporting (used by web UI)
  - structlog logging at each lifecycle stage

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

### 6. CLI (`furrow start` / `furrow web`)
- **Python CLI entry point** (`furrow/cli/main.py`) with Click.
- `furrow start "<goal>"` — run the orchestrator in the terminal.
- `furrow web` — start the FastAPI web server with a WebSocket-based real-time UI.

### 7. Web UI (`furrow/web/server.py`)
- **FastAPI app** with a WebSocket endpoint (`/ws`).
- Sends real-time JSON events (cycle start, plan, task completed/failed, test results) to the browser.
- Styled dark-mode UI that displays parallel agent activity as it happens.

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
furrow/               # Python package (mirrors the same architecture in code)
  __init__.py          # Exports + configure_logging()
  config.py            # Settings, models (Provider, TaskModel, Plan, TestResult)
  llm.py               # LLMClient (Anthropic/OpenAI/Ollama + retry + logging)
  agents/
    planner.py         # PlannerAgent
    worker.py          # WorkerAgent
    tester.py          # TesterAgent
    prompts.py         # System prompts
  core/
    orchestrator.py    # Orchestrator (core loop)
  cli/
    main.py            # Click CLI
  web/
    server.py          # FastAPI + WebSocket web UI
kilo.json              # Project config (default agent = furrow, open permissions)
AGENTS.md              # Agent instructions for coding agents
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Configuration
Settings are loaded from environment variables with the `FURROW_` prefix, or from a `.env` file.

| Setting | Default | Description |
|---|---|---|
| `FURROW_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, or `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Planner-specific model |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Worker-specific model |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Tester-specific model |
| `FURROW_MAX_CYCLES` | `0` | Max loop iterations (0 = infinite) |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Concurrent worker limit |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `FURROW_LOG_LEVEL` | `INFO` | Logging level |

## Usage
- Run `/furrow <goal>` to start the Kilo-native loop.
- Run `furrow start "<goal>"` to start the Python loop in the terminal.
- Run `furrow web` to start the web UI (default: http://localhost:8000).
- The agent runs until the goal is complete or the user stops the session (Ctrl+C).
- Mid-loop user input is incorporated into the next cycle's plan.
