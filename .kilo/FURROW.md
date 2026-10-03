# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs a **plan → execute → test → repeat** loop for
web/CLI/desktop projects. Each cycle plans tasks, executes them in parallel waves, runs the
project's test runner, and feeds the results into the next plan.

This codebase exists in two forms that share one architecture:

- **Kilo agents** (`.kilo/agent/`) for interactive TUI use.
- **The `furrow` Python package** — a standalone CLI (`furrow start` / `furrow plan` /
  `furrow web`) plus a web console, for non-TUI use.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the loop. Receives goals, plans, delegates, tests, and repeats.
- **Implementation:** `furrow/core/orchestrator.py`
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents, grouped into dependency-ordered waves
  4. Test via `furrow-tester` subagent
  5. Report and repeat

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Implementation:** `furrow/agents/planner.py` (prompts in `furrow/agents/prompts.py`)
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`,
  plus a `rationale`. Plans are normalised: blank descriptions dropped, duplicate ids
  re-numbered, dangling dependencies removed, capped at `MAX_TASKS = 5`. One unparseable
  response triggers a single repair request before the cycle fails.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Implementation:** `furrow/agents/worker.py`
- **Constraint:** 1-3 tool call rounds per task. The prompt includes the task, its files,
  its dependencies, and a truncated repository file listing.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs the project's test runner, then has the LLM summarise the output.
- **Implementation:** `furrow/agents/tester.py`
- **Verdict:** The runner's **exit code is authoritative**; the model's verdict is ignored.
  If no runner is available, tests count as passing. A run that times out is killed and
  reported as failed.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

### 6. LLM client
- **Implementation:** `furrow/llm.py`
- **Providers:** `anthropic`, `openai`, and `ollama` (via its OpenAI-compatible endpoint).
  Clients are built lazily, retried with exponential backoff up to `llm_max_retries`, and
  JSON responses are recovered from fenced or prose-wrapped payloads.

### 7. Event stream
- **Implementation:** `furrow/core/events.py`
- **Types:** `cycle_start`, `plan`, `task_start`, `task_end`, `test_start`, `test_end`,
  `cycle_end`, `error`, `done`. The CLI renders these as coloured progress lines; the web
  console streams them over a websocket.

### 8. CLI (`furrow`)
- **Implementation:** `furrow/cli/main.py` (`furrow start`, `furrow plan`, `furrow web`,
  `furrow version`)
- **`furrow plan "<goal>"`** is the dry run: one planner call, prints the task table and
  rationale, and executes nothing.
- **Shared options:** `--model`, `--provider` (`anthropic|openai|ollama`), `--workspace`,
  `--cycles` (`0` means unlimited), `--max-parallel`, `--test-command`, `--test-timeout`.
- **`furrow web`** takes `--host` (default `0.0.0.0`) and `--port` (default `8000`).
- Overrides are folded into a per-invocation settings copy, so the global settings
  singleton is never mutated.

### 9. Web console (`furrow/web/server.py`)
- Serves a single page console at `/`, plus `GET /api/health`, `GET /api/state`,
  `POST /api/run` (202; 409 if busy; 422 on a blank goal), `POST /api/stop` (404 when
  idle), and the `WS /ws` event stream.
- **One run is served at a time**: the REST and websocket paths share a single global
  registry slot.

## Loop termination
The loop stops for exactly four reasons:

1. `request_stop()` — Ctrl+C, or the web Stop button. Honoured between cycles.
2. `max_cycles` when positive. `max_cycles <= 0` (the default) means **unlimited**.
3. Goal complete — every produced task is `completed` and no test failure is outstanding.
4. No-progress guard — after **3 consecutive cycles** with no new tasks and no state change.

Failing tests do not stop the run; the failures are folded into the next planner prompt.

## Configuration
All `Settings` fields in `furrow/config.py` are overridable via **`FURROW_`-prefixed
environment variables** or a **`.env`** file: `provider`, `model`, `planner_model`,
`worker_model`, `tester_model`, `anthropic_api_key`, `openai_api_key`, `ollama_base_url`,
`max_parallel_tasks`, `max_cycles`, `workspace`, `max_tokens`, `request_timeout`,
`llm_max_retries`, `test_command`, `test_timeout`, `log_level`.

`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` must be set, or a local Ollama endpoint must be
reachable.

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
  config.py            # Settings, Plan, TaskModel, TestResult
  llm.py               # Provider clients, JSON extraction, retries
  core/
    orchestrator.py    # The plan -> execute -> test loop
    events.py          # Event / EventType
  agents/
    planner.py         # Goal -> normalised plan
    worker.py          # Single task execution
    tester.py          # Test runner discovery and verdict
    prompts.py         # Prompt templates
  cli/main.py          # start / plan / web / version
  web/server.py        # FastAPI console, REST API, websocket stream
tests/                 # pytest suite
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop in the TUI.
- Or use the CLI: `furrow start "<goal>"`, `furrow plan "<goal>"` (dry run),
  `furrow web` (live console), `furrow version`.
- The agent runs until the goal is complete, the cycle budget is reached, you stop it, or
  the no-progress guard fires.
- Mid-loop user input is incorporated into the next cycle's plan.

## Future Work
- Persistent state file for goal/task tracking across sessions.
- Desktop wrapper for non-TUI interaction.