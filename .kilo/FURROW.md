# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects. It can be used as a Kilo agent (via `/furrow`) or as a standalone Python package (`furrow start` / `furrow web`).

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
- **Output:** JSON with `summary`, `files_changed`, and `edits[]` (each edit has `path`, `old_content`, `new_content`). The orchestrator applies edits to disk after the worker returns.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Returns pass/fail JSON with a summary and failure list.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

## Python Package

The standalone Python package mirrors the Kilo agent workflow:

| File | Role |
|---|---|
| `furrow/core/orchestrator.py` | Orchestrator — the infinite loop with structured logging, cycle history, max_cycles enforcement, and state persistence |
| `furrow/agents/planner.py` | PlannerAgent — calls LLM, parses JSON into `Plan` |
| `furrow/agents/worker.py` | WorkerAgent — calls LLM, parses JSON into `WorkerResult`, writes files to disk |
| `furrow/agents/tester.py` | TesterAgent — runs test suites, evaluates results |
| `furrow/llm.py` | LLMClient — dispatches to Anthropic, OpenAI, or Ollama |
| `furrow/config.py` | Data models: `Provider`, `TaskModel`, `Plan`, `TestResult`, `FileEdit`, `WorkerResult`, `Settings` |
| `furrow/cli/main.py` | Click CLI with `start` and `web` commands |
| `furrow/web/server.py` | FastAPI app with WebSocket streaming for real-time progress |
| `tests/test_core.py` | Test suite for models, worker, orchestrator logic, and Ollama provider |

## State Persistence

After each cycle (when not done), the orchestrator saves state to `.furrow_state.json`:
- Current goal
- Cycle count
- Full cycle history (plans, task statuses, test results)
- The latest plan

## Termination Logic

The loop terminates when any of these conditions is met:
1. All tasks in the current plan are completed and tests pass
2. `max_cycles` is reached (hard limit; even failing tests won't extend past it)
3. The planner returns no tasks (goal is deemed complete)

If tests fail but tasks are complete, the loop continues with a refocused goal: `"Fix failing tests:\n" + failures`.

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
furrow/                # Python package
tests/                 # Test suite
.env.example           # Example environment configuration
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop (Kilo agent mode).
- Run `furrow start "<goal>"` to start the loop (standalone CLI mode).
- Run `furrow web` to launch the web UI with real-time WebSocket streaming.
- The agent runs until the goal is complete or the user interrupts (Ctrl+C).
