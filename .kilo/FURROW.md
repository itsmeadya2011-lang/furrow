# Furrow Agent Architecture

## Overview

Furrow is an autonomous coding agent that runs a development loop for web/CLI projects. The loop stops when tests pass, when `max_cycles` is reached (`0` means unlimited), or after 3 consecutive cycles with no worker file writes.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the planner -> worker -> tester loop until a stop condition is met. Accumulates task state across cycles.
- **Loop:**
   1. Emit `CYCLE_START`
   2. Call the planner to produce a `Plan`
   3. Execute tasks in dependency-respecting waves, up to `max_parallel_tasks` concurrent workers
   4. Call the tester to judge the output
   5. If tests passed, stop. If `max_cycles` or no-progress limit reached, stop. Otherwise continue.

### 2. Event bus (`furrow/core/events.py`)
- **Type:** Library module
- **Role:** JSON-safe progress-event emitter. The orchestrator emits typed events (`RUN_START`, `CYCLE_START`, `PLAN`, `TASK_START`, `TASK_END`, `TEST`, `CYCLE_END`, `RUN_END`, `LOG`, `ERROR`) which are fanned out to rich console subscribers and the web layer. This is what lets the web UI observe a run that previously only wrote to a console.

### 3. Planner (`furrow/agents/planner.py`)
- **Type:** Subagent (LLM-backed)
- **Role:** Breaks a high-level goal into 1–5 parallelizable tasks. Deduplicates ids and drops dangling dependencies. An empty task list is an error.

### 4. Worker (`furrow/agents/worker.py`)
- **Type:** Subagent (LLM-backed)
- **Role:** Implements one task by writing real files into the workspace. Returns a `WorkerResult` JSON with `files[].path` / `files[].content` / `files[].action`. Rejects absolute paths, `..` traversal and symlink escapes.

### 5. Tester (`furrow/agents/tester.py`)
- **Type:** Subagent (LLM-backed)
- **Role:** Discovers and runs **one** test runner (pytest, npm, cargo, go, or an explicit `test_command`). Judges its output and returns a `TestResult` JSON. Lint/type output that appears inside the captured test output is passed to the model for judgment, but no separate tool is invoked.

### 6. CLI (`furrow/cli/main.py`)
- **Type:** Click entry point
- **Role:** `furrow start` runs the loop; `furrow web` serves the monitoring UI.

### 7. Web UI (`furrow/web/server.py`)
- **Type:** FastAPI + WebSocket
- **Role:** Serves a self-contained HTML page and streams progress events. Supports concurrent runs, history replay and stop.

## Files

```
furrow/
  __init__.py
  agents/
    __init__.py
    planner.py
    prompts.py
    tester.py
    worker.py
  cli/
    __init__.py
    main.py
  config.py
  core/
    __init__.py
    events.py
    orchestrator.py
  jsonutil.py
  llm.py
  models.py
  web/
    __init__.py
    server.py
```

## Permissions

- The worker rejects absolute paths, `..` traversal and symlink escapes, so a model cannot write outside the workspace.
- The web UI binds `127.0.0.1` by default and warns on any other host because it is unauthenticated and drives an agent with file-write and subprocess access.

## Usage

- Run `python -m furrow.cli.main start "goal"` to start the loop.
- The loop stops when tests pass, when `max_cycles` is reached, or after 3 consecutive cycles with no file writes.
- Mid-loop user input is incorporated into the next cycle's plan.
- Run `python -m furrow.cli.main web` to open the monitoring UI.

## Future Work

- Persistent state file for goal/task tracking across sessions. Currently all state is in-memory per `Orchestrator` instance; nothing is persisted across process restarts.
- Desktop wrapper (the CLI already covers non-TUI interaction via `furrow start` / `furrow web`).
