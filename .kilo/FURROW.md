# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects. Supports Anthropic, OpenAI, and Ollama providers.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the infinite loop. Receives goals, plans, delegates, tests, and repeats.
- **State:** Tracks `last_plan` for the current session; goal is appended with test failure details on retry.
- **Cycle limit:** Enforces `max_cycles` (from config) before halting.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Append failures to goal and repeat

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Context:** Reads files from the configured workspace directory for file context.
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
furrow/
  __init__.py
  llm.py               # LLM client with Anthropic/OpenAI/Ollama support
  config.py            # Settings, models, provider enum
  agents/
    __init__.py
    planner.py         # Planner agent implementation
    worker.py          # Worker agent implementation
    tester.py          # Tester agent implementation
    prompts.py         # Shared prompts
  core/
    __init__.py
    orchestrator.py    # Main orchestrator loop
  cli/                 # CLI entry point
  web/                 # Web server for monitoring
kilo.json              # Project config (default agent = furrow, open permissions)
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.
Configurable via `Settings`: provider, model, planner/worker/tester models, request timeout, max parallel tasks, max cycles, workspace, log level. Env prefix `FURROW_`, optional `.env` file.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## Future Work
- CLI and desktop wrappers for non-TUI interaction.
- Persistent state file for goal/task tracking across sessions *(in-memory tracking via `last_plan` exists; not yet persisted)*.
- Web UI for monitoring parallel agent activity *(basic server implemented; streaming progress endpoints pending)*.
- Retry logic with backoff for LLM calls.
- Robust JSON parsing with fallback recovery for planner/tester output.
- Configurable request timeouts per provider.
- Structured logging across agents.
