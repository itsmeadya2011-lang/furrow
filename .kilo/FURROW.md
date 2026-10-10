# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for
web/CLI/desktop projects. It exists in two forms:

1. **Kilo agent system** (`.kilo/`) — Agent personas run by the Kilo runtime.
2. **Python package** (`furrow/`) — A standalone async implementation with CLI
   and web interfaces.

## Components

### 1. Orchestrator (`furrow`)
- **Kilo:** Type `primary`, color `#4CAF50`, 50 steps. Runs the infinite loop.
- **Python:** `furrow/core/orchestrator.py` — Async `Orchestrator` class.

Both run the same cycle:
1. **Receive / Identify** - Understand the current goal
2. **Plan** - Break the goal into 1-5 parallelizable tasks via the planner
3. **Execute** - Spawn parallel workers to complete each planned task
4. **Test** - Run tests, linting, and type checks; fix failures
5. **Report** - Show a summary of what was built, tested, and what remains
6. **Repeat** - Immediately start the next cycle until the goal is complete

### 2. Planner (`furrow-planner`)
- **Kilo:** Type `subagent`, color `#2196F3`, 10 steps.
- **Python:** `furrow/agents/planner.py` — `PlannerAgent.plan()` returns a
  `Plan` (pydantic model with `tasks[]` and `rationale`).

Breaks a high-level goal into parallelizable, independent tasks.
Output: JSON with `tasks[]`, each having `id`, `description`, `files`,
`dependencies`.

### 3. Worker (`furrow-worker`)
- **Kilo:** Type `subagent`, color `#FF9800`, 15 steps.
- **Python:** `furrow/agents/worker.py` — `WorkerAgent.run()` calls
  `LLMClient.complete_with_tools()` so the LLM can `read_file`, `write_file`,
  and `list_files`.

Executes a single assigned task using LLM tool-use for file I/O.
Constraint: 1-3 tool call rounds per task.

### 4. Tester (`furrow-tester`)
- **Kilo:** Type `subagent`, color `#9C27B0`, 15 steps.
- **Python:** `furrow/agents/tester.py` — `TesterAgent.run()` runs the
  project's test suite (auto-detected from manifests) and judges pass/fail.

Runs the project's test suite, linter, and type checker. Returns a `TestResult`
(pydantic model with `passed`, `summary`, `failures`).

### 5. Command (`/furrow`)
- **Kilo:** Slash command that invokes the `furrow` orchestrator with a goal
  from `$ARGUMENTS`.
- **Python:** `furrow start "<goal>"` (CLI) and `furrow web` (FastAPI + WebSocket
  TUI for browser-based monitoring).

## Files
```
.kilo/
  FURROW.md                 # This file
  agent/
    furrow.md               # Orchestrator (Kilo)
    furrow-planner.md       # Planner subagent
    furrow-worker.md        # Worker subagent
    furrow-tester.md        # Tester subagent
  command/
    furrow.md               # /furrow slash command
kilo.json                   # Project config (default agent = furrow)
furrow/                     # Python package
  config.py                 # Settings, models, Provider enum
  llm.py                    # LLMClient (Anthropic/OpenAI/Ollama + tools)
  agents/
    prompts.py              # PLANNER/WORKER/TESTER prompt strings
    planner.py              # PlannerAgent
    worker.py               # WorkerAgent (file editing via tool-use)
    tester.py               # TesterAgent (test runner detection)
  core/
    orchestrator.py         # Orchestrator loop (state tracking, max_cycles)
  cli/
    main.py                 # Click CLI: start, web
  web/
    server.py               # FastAPI app with WebSocket streaming
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- **Kilo:** Run `/furrow <goal>` to start the loop.
- **Python CLI:** `furrow start "<goal>"`
- **Python Web:** `furrow web` then open `http://localhost:8000`

The agent runs until the goal is complete, `max_cycles` is reached, or the
user stops the session.

## Future Work
- CLI and desktop wrappers for non-TUI interaction.
- Persistent state file for goal/task tracking across sessions.
- Web UI for monitoring parallel agent activity.
