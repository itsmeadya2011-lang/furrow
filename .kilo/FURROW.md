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
  3. Execute via parallel workers (dependency-respecting)
  4. Test via tester agent
  5. Report and repeat
- **Features:**
  - Dependency-respecting task execution (tasks run in dependency-ordered waves)
  - Cycle limit via `--max-cycles` or `FURROW_MAX_CYCLES`
  - State persistence to `.furrow_state.json` (resumable with `--resume`)
  - Structured logging via `structlog`
  - Retry logic for LLM calls via `tenacity`

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Capability:** Reads existing files, writes new/modified files via LLM-generated file operations.
- **Constraint:** 1-3 tool call rounds per task.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Fixes failures. Returns pass/fail JSON.
- **Supported runners:** pytest, npm, pnpm, yarn, cargo, go

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
furrow/
  config.py            # Pydantic settings, data models, state persistence
  llm.py               # LLM client (Anthropic, OpenAI, Ollama + retries)
  core/orchestrator.py # Main development loop
  agents/planner.py    # Task planner
  agents/worker.py     # Task executor with file I/O
  agents/tester.py     # Test runner and evaluator
  agents/prompts.py    # Agent system prompts
  cli/main.py          # Click CLI (start, web commands)
  web/server.py        # FastAPI web UI with WebSocket streaming
tests/                 # Test suite
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete, max cycles reached, or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.
- State is saved after each cycle; use `--resume` to continue from a saved state.

## Future Work
- Persistent state file for goal/task tracking across sessions (implemented)
- Web UI for monitoring parallel agent activity (implemented)
- Ollama provider support (implemented)
- CLI and desktop wrappers for non-TUI interaction
