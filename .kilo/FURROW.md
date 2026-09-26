# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent); runnable via CLI (`furrow start`) or Web UI (`furrow web`).
- **Role:** Runs the development loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents (batched at `max_parallel_tasks`)
  4. Test via `furrow-tester` subagent
  5. Report and repeat
- **Completion:** The loop stops when all tasks complete, tests pass, the planner returns no tasks, or `max_cycles` is reached.
- **Event streaming:** Emits async events (`cycle_start`, `plan`, `task_completed`, `test_passed`, etc.) via an `on_event` callback for UI integration.

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
- `subtask: false` so it runs as a top-level task.

## Files

### Python implementation
```
furrow/
  config.py        — Settings (pydantic BaseSettings), data models (TaskModel, Plan, TestResult)
  llm.py           — LLMClient: provider routing (Anthropic / OpenAI / Ollama),
                     file I/O helpers, lazy client instantiation
  core/            — Orchestrator: infinite loop (plan→execute→test→report→repeat),
                     max_cycles enforcement, event streaming via on_event callback
  agents/          — PlannerAgent, WorkerAgent, TesterAgent (async LLM wrappers)
  cli/             — Click CLI: `furrow start <goal> [--model] [--max-cycles]`, `furrow web`
  web/             — FastAPI app: HTML UI + WebSocket streaming of orchestrator events
```

### Kilo agent/command definitions
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

## Future Work
- Persistent state file for goal/task tracking across sessions.
- Desktop app wrapper for non-TUI interaction.
