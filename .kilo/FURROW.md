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
- **State:** Persists goal, cycle count, and task states to `furrow_state.json` after each cycle (see Configuration).

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
  cli/main.py          # CLI entrypoint (`furrow start`, `furrow web`)
  config.py            # Settings (env vars, .env file)
  core/orchestrator.py # Orchestrator loop, state persistence
  llm.py               # LLM client (Anthropic / OpenAI / Ollama)
  web/server.py        # Web UI + WebSocket progress streaming
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## Configuration
Settings are defined in `furrow/config.py` using `pydantic-settings`. They are loaded from (in order of precedence): CLI flags, environment variables (`FURROW_*`), and a `.env` file.

Key settings:
- `provider` — `anthropic`, `openai`, or `ollama`
- `model`, `planner_model`, `worker_model`, `tester_model` — per-agent model overrides
- `anthropic_api_key`, `openai_api_key` — API keys (or use `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` env vars)
- `ollama_base_url` — Ollama endpoint (default `http://localhost:11434`)
- `max_parallel_tasks` — max concurrent worker tasks per cycle (default `5`)
- `max_cycles` — max cycles before halting (default `0` = unlimited)
- `log_level` — structured logging level (default `INFO`)
- `state_file` — path to persistent state file (default `furrow_state.json`)
- `workspace` — working directory

## Web Server (Progress Streaming)
`furrow web` starts a FastAPI + Uvicorn server (default `http://0.0.0.0:8000`).

- `GET /` — serves a minimal HTML page with a goal input form.
- `WS /ws` — WebSocket endpoint that accepts a JSON message `{ "goal": "...", "model": "..." }`, constructs an `Orchestrator`, and streams live progress events to the client as the run executes.

This enables monitoring parallel agent activity from a browser without requiring a TUI session.

## Future Work
- CLI and desktop wrappers for non-TUI interaction.
- ~~Persistent state file for goal/task tracking across sessions.~~ **Done** — `furrow_state.json`.
- ~~Web UI for monitoring parallel agent activity.~~ **Done** — `furrow web` with WebSocket streaming.