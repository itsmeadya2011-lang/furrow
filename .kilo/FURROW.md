# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs a continuous development loop for web/CLI/desktop projects. It is implemented as a Python package (`furrow/`) with a Kilo agent front-end (`.kilo/agent/`) that drives the loop.

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
- **Python implementation:** `furrow/core/orchestrator.py` — `Orchestrator` class
  - **Cycle termination (`_is_done()` / `_get_tasks()`):** Fixed bug where the loop always exited after the first cycle. Now correctly checks that all tasks are completed, none failed, and the last test result passed.
  - **Cycle limits (`max_cycles`):** Configurable ceiling on total cycles. `0` = unlimited (default). Enforced in `run()` via `self.client.settings.max_cycles`. When exceeded, the loop stops with status `"failed"`.
  - **Parallelism limits (`max_parallel_tasks`):** Enforced via `asyncio.Semaphore`. Configurable; defaults to 5. Bounds the number of concurrent worker subagents.
  - **Worker retry logic:** Workers retry up to `MAX_WORKER_RETRIES` (2) times. On failure, fix instructions (including the original error) are appended to the task description before re-attempting.
  - **State persistence:** Saves state after each cycle to `.furrow-state.json` in the workspace. Resumes from a previous `"running"` state on startup (restoring goal and cycle count). State transitions: `"running"` → `"completed"` (goal done) or `"failed"` (max_cycles exceeded).
  - **Structured logging:** Uses `structlog` for structured log events (`cycle_start`, `task_completed`, `tests_passed`, etc.).

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.
- **Python implementation:** `furrow/agents/planner.py` — `PlannerAgent` class
  - Uses `LLMClient.complete()` (no tool-calling, planning only)
  - Parses the LLM JSON response into a `Plan` (pydantic model)

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.
- **Python implementation:** `furrow/agents/worker.py` — `WorkerAgent` class
  - **Tool-calling:** Now uses `LLMClient.complete_with_tools()` instead of `complete()`. Workers actually make code changes via three tools: `read_file`, `write_file`, `list_files`.
  - **Iteration cap:** `max_iterations` parameter (default 10) bounds the agent loop.
  - **System prompt:** Includes tool documentation and workflow guidance (see `furrow/agents/prompts.py`).
  - **Structured logging:** Logs task start, completion, and failures via `structlog`.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Fixes failures. Returns pass/fail JSON.
- **Python implementation:** `furrow/agents/tester.py` — `TesterAgent` class
  - Auto-detects the project's test runner (pytest, npm, pnpm, yarn, cargo, go)
  - Runs tests with a 120-second timeout; kills on timeout
  - Parses the LLM response into a `TestResult` (pydantic model)

### 5. LLM Client (`furrow.llm`)
- **Role:** Unified async LLM interface with provider-specific implementations.
- **Implementation:** `furrow/llm.py` — `LLMClient` class
  - `complete()`: Plain text completion (used by Planner and Tester)
  - `complete_with_tools()`: Tool-calling with agent loop (used by Worker)
  - **Three tools available to workers:** `read_file`, `write_file`, `list_files`
  - **Retry policy:** Tenacity `@retry` with 3 attempts and exponential backoff on connection errors, rate limits, and timeouts
  - **Provider support:** Separate code paths for Anthropic (`AsyncAnthropic`) and OpenAI (`AsyncOpenAI`)
  - **Structured logging:** Debug-level log events for API calls and tool executions

### 6. State Persistence (`furrow.core.state`)
- **Role:** Cross-session goal/task/test tracking.
- **Implementation:** `furrow/core/state.py` — `FurrowState` (pydantic model)
  - Saves to `.furrow-state.json` in the workspace (location configurable via `workspace` setting)
  - **Tracked fields:** `goal`, `cycles`, `tasks` (`TaskModel[]`), `test_results` (`TestResult[]`), `last_updated`, `status` (`"running"` / `"completed"` / `"failed"`)
  - **Methods:** `load()`, `save()`, `add_task()`, `update_task()`, `add_test_result()`, `set_status()`, `set_goal()`
  - **Graceful error handling:** Missing file, corrupt JSON, validation errors, and I/O errors all fall back to defaults with a warning (never crashes the loop)
  - **Cross-session resume:** On startup, if a previous state exists with `status == "running"`, the orchestrator resumes from the saved goal and cycle count

### 7. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.
- **Python CLI:** `furrow/cli/main.py`
  - `furrow start <goal>` — Start the development loop (with optional `--model` override)
  - `furrow web` — Launch a FastAPI web UI with WebSocket streaming (`furrow/web/server.py`)

## Config (`furrow/config.py`)
- **Settings** (pydantic `BaseSettings`, env-prefixed `FURROW_`):
  - `provider`: `anthropic` | `openai` | `ollama` (default: `anthropic`)
  - `model`, `planner_model`, `worker_model`, `tester_model`: LLM model selection per role
  - **`max_cycles`:** int, default `0` (unlimited). Now **enforced** by the orchestrator — stops the loop when the limit is reached.
  - **`max_parallel_tasks`:** int, default `5`. Now **enforced** via `asyncio.Semaphore` in the orchestrator.
  - **`workspace`:** Path (default: current working directory). Also determines the **state file location** — the `.furrow-state.json` state file is saved here.
  - `anthropic_api_key`, `openai_api_key`, `ollama_base_url`
  - `log_level`
- **Pydantic models:** `TaskModel`, `Plan`, `TestResult`

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
  FURROW.md            # This file
kilo.json              # Project config (default agent = furrow, open permissions)
furrow/
  __init__.py
  config.py            # Settings, TaskModel, Plan, TestResult
  llm.py               # LLMClient with tool-calling, retry, structured logging
  cli/
    main.py            # Click CLI: furrow start, furrow web
  core/
    __init__.py
    orchestrator.py    # Orchestrator with max_cycles, max_parallel_tasks, retry, state
    state.py           # FurrowState persistence model
  agents/
    __init__.py
    planner.py         # PlannerAgent
    worker.py          # WorkerAgent (tool-calling via complete_with_tools)
    tester.py         # TesterAgent
    prompts.py         # PLANNER_PROMPT, WORKER_PROMPT, TESTER_PROMPT
  web/
    __init__.py
    server.py          # FastAPI web UI with WebSocket streaming
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` in Kilo to start the loop.
- Or use the Python CLI: `furrow start "<goal>"`.
- The agent runs until the goal is complete, `max_cycles` is reached, or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.
- State persists across sessions in `.furrow-state.json`; a previous `"running"` state is resumed automatically.
- Launch the web UI with `furrow web` (or `/furrow` with the web wrapper).

## Future Work
- [completed] Persistent state file for goal/task tracking across sessions.
- [completed] Web UI for monitoring agent activity (basic HTTP + WebSocket, `furrow/web/server.py`).
- [completed] CLI wrapper for non-TUI interaction (`furrow/cli/main.py`).
- Desktop wrappers for non-TUI interaction (native GUI still pending).
- Rich real-time monitoring dashboard (beyond the current minimal web page).
- Multi-provider support beyond Anthropic and OpenAI (Ollama provider scaffolded but not fully wired).
