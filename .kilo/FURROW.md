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

## Future Work
- CLI and desktop wrappers for non-TUI interaction.
- Persistent state file for goal/task tracking across sessions.
- Web UI for monitoring parallel agent activity.

## Architecture Details

### LLM Client (`furrow/llm.py`)
- **Providers:** Anthropic, OpenAI, and Ollama (via HTTP API).
- **Tool-calling:** `LLMClient` exposes a `run_tool_loop()` method that repeatedly
  calls the LLM with a list of `Tool` objects until no more tool calls are requested.
  Each `Tool` has a name, description, JSON-schema parameters, and an async function
  backing implementation.
- Built-in tools available to workers: `read_file`, `write_file`, `list_files`,
  `run_command`.

### Orchestrator (`furrow/core/orchestrator.py`)
- **Cycle:** Plan → Execute (parallel workers) → Test → decide whether to continue.
- **Termination:** The loop stops when the goal is complete (all tasks done and
  tests pass) or when `max_cycles` is reached (0 = unlimited).
- **State:** Tracks `current_plan` across cycles for introspection. Failed tests
  trigger a focused retry cycle with an updated goal.

### Worker (`furrow/agents/worker.py`)
- Receives a single task and runs a tool-calling loop using the shared `LLMClient`.
- Exposes file-system and command-execution tools backed by `LLMClient` methods.

### Tester (`furrow/agents/tester.py`)
- Runs the project's test suite (pytest, npm, cargo, go, etc.) in the workspace
  directory and delegates pass/fail evaluation to the LLM.

### Settings (`furrow/config.py`)
- All configuration via environment variables with `FURROW_` prefix or `.env` file.
- `max_cycles` controls the loop limit (0 = unlimited).
