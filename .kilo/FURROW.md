# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the infinite loop. Receives goals, plans, delegates, tests, and repeats.
- **State:** Maintains `OrchestratorState` tracking cycle history, failed tasks, and goal modifications.
- **Loop:**
   1. Receive / Identify goal
   2. Plan parallel tasks (1-5)
   3. Execute via parallel `furrow-worker` subagents
   4. Test via `furrow-tester` subagent
   5. Report and repeat
- **Batching:** Splits tasks into batches of `max_parallel_tasks` to avoid overwhelming the system.
- **Retry:** Wraps planner and tester calls with tenacity retry logic on transient failures.

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.
- **Retry:** Retries JSON parsing once on failure with a simplified prompt.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.
- **Context:** Reads files listed in the task to understand existing code before editing.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Fixes failures. Returns pass/fail JSON.
- **Output:** `TestResult` with `passed`, `summary`, and `failures` fields.
- **Runners:** Auto-detects pytest, npm, cargo, go, and other common test commands.

### 5. LLM Client (`furrow-llm`)
- **Type:** Utility
- **Role:** Unified async client for Anthropic, OpenAI, and Ollama.
- **Retry:** Exponential backoff for rate limits, connection errors, timeouts, and 5xx status codes.

### 6. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

## Configuration

| Setting | Description | Default |
|---|---|---|
| `FURROW_PROVIDER` | LLM provider | `anthropic` |
| `FURROW_MODEL` | Default model | `claude-sonnet-4-20250514` |
| `FURROW_PLANNER_MODEL` | Planner model | `claude-3-5-haiku-20241022` |
| `FURROW_WORKER_MODEL` | Worker model | `claude-3-5-sonnet-20241022` |
| `FURROW_TESTER_MODEL` | Tester model | `claude-3-5-sonnet-20241022` |
| `FURROW_ANTHROPIC_API_KEY` | Anthropic key | |
| `FURROW_OPENAI_API_KEY` | OpenAI key | |
| `FURROW_OLLAMA_BASE_URL` | Ollama URL | `http://localhost:11434` |
| `FURROW_MAX_PARALLEL_TASKS` | Parallel task limit | `5` |
| `FURROW_MAX_CYCLES` | Cycle limit (0 = unlimited) | `0` |
| `FURROW_MAX_RETRIES` | LLM retry attempts | `3` |
| `FURROW_WORKSPACE` | Workspace directory | `.` |
| `FURROW_LOG_LEVEL` | Log level | `INFO` |
| `FURROW_MAX_TOKENS` | Max tokens per completion | `4096` |
| `FURROW_TEST_COMMANDS` | Test runner commands | Auto-detected |

## Files
```
furrow/
  __init__.py
  config.py          # Settings, Provider enum, data models
  llm.py             # LLMClient with Anthropic/OpenAI/Ollama support
  core/
    orchestrator.py  # Orchestrator + OrchestratorState
  agents/
    planner.py       # PlannerAgent
    worker.py        # WorkerAgent
    tester.py        # TesterAgent
    prompts.py       # Agent system prompts
  cli/
    main.py          # Click CLI: start, web
  web/
    server.py        # FastAPI + WebSocket UI
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
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## Completed
- CLI wrapper: `furrow start` and `furrow web` commands.
- Persistent state tracking: `OrchestratorState` records cycles, failed tasks, and goal history.
- Retry logic: LLM calls retry with exponential backoff; planner and tester have their own retry wrappers.
- `max_cycles` and `max_parallel_tasks`: Configurable limits for cycle count and parallel workers.
- Web UI: FastAPI + WebSocket interface for monitoring.
- Ollama provider: Local LLM support added to `LLMClient`.

## Future Work
- Persistent state file for goal/task tracking across sessions.
- Enhanced failure recovery with automatic patch generation.
- Streaming token output in the Web UI.
