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

## Recent Improvements
- **Fixed orchestrator task tracking**: `_get_tasks()` now returns the latest plan's tasks instead of `[]`, enabling correct completion detection.
- **Cycle limits**: `max_cycles` setting is now enforced in the orchestrator loop.
- **Parallel task limits**: `max_parallel_tasks` setting is now enforced via `asyncio.Semaphore`, preventing unbounded concurrency.
- **LLM retry logic**: `@retry` with exponential backoff (tenacity) on Anthropic and OpenAI API calls for transient failures.
- **Structured logging**: `structlog`-based logging throughout the orchestrator, planner, worker, tester, and LLM client.
- **WebSocket streaming**: Web UI now streams orchestrator console output to the browser in real time; handles disconnections gracefully.
- **Worker file writing**: Worker agents now parse structured JSON output from the LLM and write files to disk via `LLMClient.write_file`.
- **Ollama provider**: `Provider.OLLAMA` is now fully implemented using httpx.
- **Tester robustness**: Broad `except Exception` narrowed to `RuntimeError`; distinguishes "no test runner" from genuine failures; adds structured logging.
