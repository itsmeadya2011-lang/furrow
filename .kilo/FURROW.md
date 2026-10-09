# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects. It supports CLI, web UI, and slash-command entry points.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **File:** `furrow/core/orchestrator.py`
- **Role:** Runs the infinite loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal (from user input or restored saved state)
  2. Plan parallel tasks (1-5) with project context
  3. Execute via parallel `furrow-worker` subagents, respecting task dependencies
  4. Test via `furrow-tester` subagent
  5. If tests fail, update goal to fix failures and continue
  6. Repeat until all tasks complete or `max_cycles` is reached

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **File:** `furrow/agents/planner.py`
- **Role:** Breaks a high-level goal into 1-5 parallelizable, independent tasks.
- **Context:** Receives project file listing and prior cycle task results.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **File:** `furrow/agents/worker.py`
- **Role:** Executes a single assigned task. Parses LLM output for file-write directives (JSON array or markdown code blocks with filename headers) and writes files to disk.
- **Constraint:** 1-3 tool call rounds per task.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **File:** `furrow/agents/tester.py`
- **Role:** Runs the project's test suite, linter, and type checker. Evaluates pass/fail.
- **Test runners:** pytest, npm/pnpm/yarn test, cargo test, go test.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **File:** `.kilo/command/furrow.md`
- **Role:** Entry point that invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

## Configuration

### `kilo.json` (Kilo project config)
Sets `furrow` as the default agent with open permissions (bash, edit, read all `allow`).

### `furrow.json` (Furrow project config)
Project-level settings file, created with `furrow init`. Supports:
- `max_cycles` — maximum loop iterations (0 = unlimited)
- `max_parallel_tasks` — parallel worker concurrency
- `workspace` — working directory
- `settings.provider`, `model`, etc. — LLM provider overrides

### Environment variables
All settings can be overridden via `FURROW_`-prefixed env vars (e.g., `FURROW_MODEL`, `FURROW_MAX_CYCLES`, `FURROW_PROVIDER`).

### LLM Providers
- **Anthropic** (default): `ANTHROPIC_API_KEY`
- **OpenAI**: `OPENAI_API_KEY`
- **Ollama**: `FURROW_OLLAMA_BASE_URL` (defaults to `http://localhost:11434`)

## File Layout
```
furrow/
├── __init__.py           # Package exports
├── config.py             # Pydantic models + Settings (BaseSettings)
├── llm.py                # LLMClient (Anthropic/OpenAI/Ollama + filesystem helpers)
├── cli/
│   └── main.py           # Click CLI: `furrow start`, `furrow web`, `furrow init`
├── core/
│   └── orchestrator.py   # Orchestrator - runs the infinite loop
├── agents/
│   ├── planner.py        # PlannerAgent - breaks goals into tasks
│   ├── worker.py         # WorkerAgent - executes tasks, writes files
│   ├── tester.py         # TesterAgent - runs tests, evaluates results
│   └── prompts.py        # Prompt templates
└── web/
    └── server.py         # FastAPI web UI with WebSocket streaming

.kilo/
├── FURROW.md             # This file
├── agent/
│   ├── furrow.md         # Orchestrator agent definition
│   ├── furrow-planner.md # Planner subagent definition
│   ├── furrow-worker.md  # Worker subagent definition
│   └── furrow-tester.md  # Tester subagent definition
└── command/
    └── furrow.md         # /furrow slash command
```

## State Management
- State is persisted to `.furrow-state.json` in the workspace after each cycle.
- On startup, if the state file exists, the goal and task state are restored.
- State is automatically cleaned up when the goal is fully complete.

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` in the Kilo TUI to start the loop.
- Run `furrow start "<goal>"` from the CLI.
- Run `furrow web` to start the web UI on port 8000.
- The agent runs until the goal is complete, `max_cycles` is reached, or the user stops the session.

## Logging
Uses `structlog` with a `ConsoleRenderer` for structured, readable log output.
Log level is configurable via `FURROW_LOG_LEVEL` or `furrow.json`.
