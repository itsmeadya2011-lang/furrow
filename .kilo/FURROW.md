# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects. It supports two runtime modes:

1. **Kilo TUI** — Run `/furrow <goal>` inside the Kilo agent runtime. The orchestrator uses the Task tool to spawn `furrow-worker` subagents with file manipulation capabilities.
2. **Standalone CLI** — Run `furrow start "<goal>"` or `furrow web` for a web UI. The Python package calls LLM APIs directly through an abstraction layer (Anthropic, OpenAI, Ollama).

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the infinite loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5, respecting dependencies and `max_parallel_tasks`)
  3. Execute via parallel `furrow-worker` subagents (Kilo TUI) or parallel `WorkerAgent` coroutines (CLI)
  4. Test via `furrow-tester` subagent (Kilo TUI) or `TesterAgent` (CLI)
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

## Python Package (`furrow/`)

### Core Modules
| File | Description |
|------|-------------|
| `furrow/config.py` | Settings (pydantic BaseSettings), TaskModel, Plan, TestResult |
| `furrow/llm.py` | LLMClient — async abstraction over Anthropic, OpenAI, Ollama with retry logic |
| `furrow/core/orchestrator.py` | Orchestrator — runs the plan→execute→test→repeat loop |
| `furrow/agents/planner.py` | PlannerAgent — sends plan prompt to LLM, parses JSON Plan |
| `furrow/agents/worker.py` | WorkerAgent — executes a single task via LLM |
| `furrow/agents/tester.py` | TesterAgent — runs test suite + sends results to LLM |
| `furrow/agents/prompts.py` | Prompt templates for planner, worker, tester |
| `furrow/cli/main.py` | Click CLI: `furrow start` and `furrow web` |
| `furrow/web/server.py` | FastAPI web UI with WebSocket streaming |

### Configuration
Settings are configurable via environment variables (prefixed `FURROW_`) or `.env` file:

| Setting | Default | Description |
|---------|---------|-------------|
| `provider` | `anthropic` | LLM provider: `anthropic`, `openai`, `ollama` |
| `model` | `claude-sonnet-4-20250514` | Default LLM model |
| `planner_model` | `claude-3-5-haiku-20241022` | Model for planning |
| `worker_model` | `claude-3-5-sonnet-20241022` | Model for task execution |
| `tester_model` | `claude-3-5-sonnet-20241022` | Model for testing |
| `max_parallel_tasks` | `5` | Max concurrent worker tasks per cycle |
| `max_cycles` | `0` | Max cycles (0 = infinite) |
| `workspace` | `cwd` | Working directory |
| `log_level` | `INFO` | Logging level |

### Features
- **Retry Logic:** LLM API calls retry up to 3 times with exponential backoff via `tenacity`. `ValueError` (programming errors) fail fast.
- **Dependency Resolution:** Tasks with `dependencies` are executed only after their dependencies complete. Tasks are executed in waves respecting `max_parallel_tasks`.
- **Structured Logging:** All components use `structlog` for structured logging, configurable via `FURROW_LOG_LEVEL`.
- **Ollama Support:** The `ollama` provider uses the OpenAI-compatible API via the `ollama_base_url` setting.

## Files
```
.kilo/
  agent/
    furrow.md          # Orchestrator agent (Kilo TUI)
    furrow-planner.md  # Planner subagent
    furrow-worker.md   # Worker subagent
    furrow-tester.md   # Tester subagent
  command/
    furrow.md          # /furrow slash command
furrow/
  __init__.py          # Package init, configure_logging()
  config.py            # Settings, models, Provider enum
  llm.py               # LLMClient with retry + multi-provider
  cli/main.py          # Click CLI entry point
  core/
    orchestrator.py    # Orchestrator - the development loop
  agents/
    planner.py         # PlannerAgent
    worker.py          # WorkerAgent
    tester.py          # TesterAgent
    prompts.py         # Prompt templates
  web/
    server.py          # FastAPI web UI
kilo.json              # Kilo project config (default_agent=furrow, open permissions)
pyproject.toml         # Python package config
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage

### Kilo TUI
```
/furrow "Add JWT authentication to the API"
```

### CLI
```bash
furrow start "Add JWT authentication to the API" --cycles 5
furrow start "Build a REST API" --provider openai --model gpt-4
furrow web --port 8080
```

### Environment Variables
```bash
export FURROW_PROVIDER=openai
export FURROW_OPENAI_API_KEY=sk-...
export FURROW_PLANNER_MODEL=gpt-4o-mini
export FURROW_WORKER_MODEL=gpt-4o
export FURROW_TESTER_MODEL=gpt-4o
export FURROW_MAX_CYCLES=10
export FURROW_LOG_LEVEL=DEBUG
```

## Testing
Run the test suite:
```bash
pytest tests/ -v
```

## Future Work
- Desktop wrapper (Tauri/Electron) for non-TUI interaction.
- Persistent state file for goal/task tracking across sessions.
- Improved web UI for monitoring parallel agent activity.
- File system sandbox for workspace isolation.
