# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the infinite loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5, respecting `max_parallel_tasks`)
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Report and repeat (up to `max_cycles` per session)
- **Key modules:**
  - `furrow/core/orchestrator.py` — main loop, task storage (`self.tasks`), done detection (`_is_done()`), cycle accounting, state persistence (`_save_state`/`_load_state`)
  - `furrow/llm.py` — provider abstraction (Anthropic, OpenAI, Ollama) with tenacity retries; exposes a module-level `logger`
  - `furrow/web/server.py` — FastAPI + WebSocket streaming server for the web UI
  - `furrow/cli/main.py` — Click CLI entry point (`furrow start`, `furrow web`)
  - `furrow/core/orchestrator.py` calls `structlog.configure()` at module import time; `llm.py` exposes a module-level `logger` via `structlog.get_logger(__name__)`.

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
furrow/
  core/
    orchestrator.py    # Loop, task storage, cycle/state enforcement
  llm.py               # Provider clients (Anthropic / OpenAI / Ollama) + retries
  web/
    server.py          # WebSocket streaming for web UI
  cli/
    main.py            # CLI entry point (`furrow start`, `furrow web`)
  config.py            # Settings, Provider enum, TaskModel, Plan, TestResult
  agents/
    planner.py         # PlannerAgent
    worker.py          # WorkerAgent
    tester.py          # TesterAgent
    prompts.py         # System prompts
.furrow/
  state.json           # Persisted goal/cycle/task state (auto-managed)
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete, `max_cycles` is reached, or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.
- State persists across cycles in `.furrow/state.json` so a restarted session can resume from the last cycle.

## Configuration
All settings are read from environment variables with the `FURROW_` prefix (via `pydantic-settings` in `furrow/config.py`).

| Variable | Default | Purpose |
|---|---|---|
| `FURROW_PROVIDER` | `anthropic` | One of `anthropic`, `openai`, `ollama`. |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model for general LLM calls. |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the planner agent. |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the worker agent. |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the tester agent. |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Maximum concurrent `furrow-worker` subagents per cycle. |
| `FURROW_MAX_CYCLES` | `0` (unlimited) | Hard cap on orchestrator loop iterations per session. |
| `FURROW_STATE_FILE` | `.furrow/state.json` | Location of the persisted state file. |
| `FURROW_WORKSPACE` | current directory | Workspace root for file operations. |
| `FURROW_LOG_LEVEL` | `INFO` | `structlog` log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `FURROW_ANTHROPIC_API_KEY` | _(from env)_ | Anthropic API key (falls back to `ANTHROPIC_API_KEY`). |
| `FURROW_OPENAI_API_KEY` | _(from env)_ | OpenAI API key (falls back to `OPENAI_API_KEY`). |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL for the Ollama provider. |

Provider credentials (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) are read by the underlying SDKs.

## Current Status
- `_get_tasks()` bug fixed — the orchestrator now correctly stores and returns `self.tasks` so `_is_done()` and reporting see the real task list.
- `max_cycles` and `max_parallel_tasks` are actually enforced (cycle counter incremented per iteration; parallelism capped per spawn).
- Ollama provider added in `llm.py` alongside Anthropic and OpenAI.
- State persistence via `.furrow/state.json` (written each cycle, reloaded on startup).
- `structlog` configured at module level in `orchestrator.py` (with `structlog.configure()`); both `orchestrator.py` and `llm.py` expose a module-level `logger`; `tenacity` retries wrap transient LLM/SDK errors.
- WebSocket streaming exposes cycle output to the web UI via `furrow/web/server.py`; the `Orchestrator` accepts a `console` parameter so the web server can inject a WebSocket-aware `rich.console.Console`.

## Future Work
- CLI and desktop wrappers for non-TUI interaction.
- Web UI: a viewer is wired up via WebSocket, but rendering polish (cycle history table, diff viewer, per-worker logs) is still pending.
- Smarter dependency-aware planning — currently the planner emits `dependencies` but the orchestrator does not yet schedule waves based on them.
