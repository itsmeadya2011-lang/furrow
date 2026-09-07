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
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## Python Implementation

Furrow also ships as a Python package (`furrow/`) with a full CLI and web server:

```
furrow/
  config.py            # Data models + Settings (pydantic-settings, structlog)
  llm.py               # LLMClient: async Anthropic/OpenAI + file/shell tools
  core/orchestrator.py # Orchestrator: the infinite plan→execute→test loop
  agents/
    planner.py         # PlannerAgent: goal → Plan (JSON, with retry)
    worker.py          # WorkerAgent: task → file changes (tool-use loop)
    tester.py          # TesterAgent: runs tests + lint, returns TestResult
  cli/main.py          # Click CLI: `furrow start`, `furrow web`
  web/server.py        # FastAPI app + WebSocket streaming UI
tests/                 # pytest test suite
```

### Configuration
- `Settings` (pydantic-settings) reads from env vars (`FURROW_*`) and `.env`.
- `max_cycles` caps the loop (default 20; set to 0 for unlimited).
- `structlog` provides structured logging; `FurrowState` supports persistence.

### Worker Tool-Use Loop
Workers call the LLM, parse JSON tool-action blocks from the response, and execute them via `LLMClient.read_file`, `write_file`, `list_files`, and `run_shell`. The loop runs up to 15 rounds.

### Web UI
The `/ws` WebSocket endpoint streams orchestrator output in real time via an `on_output` callback and `asyncio.Queue`.

## Future Work
- Desktop GUI wrapper for non-CLI interaction.
- Native LLM function-calling (Anthropic tool_use / OpenAI function calling) instead of text-based JSON tool protocol.
- Automatic context compaction for long-running sessions.
- Provider-aware default model selection (e.g., GPT-4 when `provider=OPENAI`).
