# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects.
It supports both a Kilo TUI mode (via `.kilo/agent/` definitions) and a standalone Python CLI/Web mode.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the infinite loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` agents (Python `WorkerAgent` or Kilo subagents)
  4. Test via `furrow-tester` subagent
  5. Report and repeat until all tasks complete and tests pass, or max_cycles is reached, or user stops

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Output:** JSON with `changes[]` (each containing `path` and `content`) and a `summary`. The orchestrator applies file changes automatically.
- **Constraint:** 1-3 tool call rounds per task.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Returns pass/fail JSON. If no test runner is found, treats as pass (no tests to fail).
- **Output:** JSON with `passed`, `summary`, and `failures[]`.

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
  FURROW.md            # This file
kilo.json              # Project config (default agent = furrow, open permissions)

furrow/
  config.py            # Pydantic models + Settings (Provider, TaskModel, Plan, TestResult)
  llm.py               # LLMClient (Anthropic, OpenAI, Ollama supported)
  agents/
    planner.py         # PlannerAgent — generates task plans
    worker.py          # WorkerAgent — executes tasks, writes files
    tester.py          # TesterAgent — runs tests, returns TestResult
    prompts.py         # Prompt templates for all agents
  core/
    orchestrator.py    # Orchestrator — main async loop
  cli/
    main.py            # Click CLI (start, web)
  web/
    server.py          # FastAPI + WebSocket UI
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop (Kilo TUI mode).
- Run `furrow start <goal>` to start via CLI.
- Run `furrow web` to start the web UI server.
- The agent runs until the goal is complete, max_cycles is reached, or the user stops the session.
- `--max-cycles N` flag limits the number of planning cycles (0 = unlimited).
- `--model MODEL` flag overrides the LLM model.

## State Management
- Tasks are tracked in `Orchestrator._tasks` across cycles.
- `_is_done()` checks: all tasks completed + tests passed in the most recent cycle.
- The goal is updated with test failure details for retry in the next cycle.
- Stop can be requested via `request_stop()` (used by web UI).

## Implementation Notes
- LLM client supports Anthropic, OpenAI, and Ollama providers.
- Retry with exponential backoff is applied to LLM API calls.
- Workers return structured JSON with file changes; the orchestrator applies them via `LLMClient.write_file()`.
