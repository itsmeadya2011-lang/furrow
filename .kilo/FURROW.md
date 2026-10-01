# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects. It plans, executes parallel tasks, tests, and repeats until the goal is complete.

## Components

### 1. Orchestrator (`furrow` / `furrow.core.orchestrator.Orchestrator`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the development loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5) with dependencies
  3. Execute via parallel `furrow-worker` agents, respecting dependencies and `max_parallel_tasks`
  4. Test via `furrow-tester` agent
  5. Report and repeat (up to `max_cycles`, or until goal + tests pass)
- **Features:**
  - Dependency-aware task scheduling (topological execution)
  - Configurable parallelism via `max_parallel_tasks` (semaphore-bounded)
  - Cycle limiting via `max_cycles` (0 = unlimited)
  - Structured logging via `structlog`
  - Injectable `Console` for WebSocket streaming

### 2. Planner (`furrow-planner` / `furrow.agents.planner.PlannerAgent`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.
- **Features:**
  - Retry logic for transient LLM failures (tenacity)
  - Structured logging

### 3. Worker (`furrow-worker` / `furrow.agents.worker.WorkerAgent`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.
- **Features:**
  - Retry logic for transient LLM failures (tenacity)
  - Structured logging

### 4. Tester (`furrow-tester` / `furrow.agents.tester.TesterAgent`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Returns pass/fail JSON.
- **Features:**
  - Auto-detects test runner (pytest, npm, yarn, cargo, go)
  - Graceful handling of missing test runners
  - Structured logging

### 5. LLM Client (`furrow.llm.LLMClient`)
- Supports Anthropic, OpenAI, and Ollama providers.
- Configurable `model`, `temperature`, `max_tokens`.
- Retry logic with exponential backoff (tenacity).
- Conversation history support via `messages` parameter.

### 6. Command (`/furrow`)
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
  cli/main.py          # CLI entry point (furrow start, furrow web)
  config.py            # Settings & data models
  llm.py               # LLM client (Anthropic, OpenAI, Ollama)
  core/
    orchestrator.py    # Main orchestration loop
  agents/
    planner.py         # Planning agent
    worker.py          # Worker agent
    tester.py          # Testing agent
    prompts.py         # System prompts
  web/
    server.py          # WebSocket web UI server
```

## Configuration
All settings can be configured via environment variables (prefixed with `FURROW_`) or a `.env` file:

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `provider` | `FURROW_PROVIDER` | `anthropic` | LLM provider (anthropic, openai, ollama) |
| `model` | `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model |
| `planner_model` | `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Planner model |
| `worker_model` | `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Worker model |
| `tester_model` | `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Tester model |
| `ollama_model` | `FURROW_OLLAMA_MODEL` | `llama3.1:latest` | Ollama model |
| `max_parallel_tasks` | `FURROW_MAX_PARALLEL_TASKS` | `5` | Max concurrent tasks |
| `max_cycles` | `FURROW_MAX_CYCLES` | `0` | Max cycles (0 = unlimited) |
| `max_tokens` | `FURROW_MAX_TOKENS` | `8192` | Max output tokens |
| `temperature` | `FURROW_TEMPERATURE` | `0.7` | LLM temperature |
| `retry_attempts` | `FURROW_RETRY_ATTEMPTS` | `3` | Max retry attempts |
| `retry_backoff` | `FURROW_RETRY_BACKOFF` | `1.0` | Retry backoff multiplier |

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop (Kilo agent).
- Run `furrow start "<goal>"` from CLI.
- Run `furrow web` to start the WebSocket web UI.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## CLI Options
```bash
furrow start "Add JWT auth" --model claude-3-5-sonnet-20241022 --max-cycles 10 --log-level DEBUG
furrow web --host 0.0.0.0 --port 8000
```