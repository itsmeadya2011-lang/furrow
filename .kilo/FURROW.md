# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite (or bounded) development loop
for web/CLI/desktop projects. It plans parallel tasks, executes them concurrently via
subagents, tests the results, and repeats until the goal is complete or stopped.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent
- **Role:** Runs the development loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Report and repeat
- **Termination:** The loop stops when (a) all tasks are completed and tests pass,
  (b) `max_cycles` is reached (when `FURROW_MAX_CYCLES > 0`; default 0 = infinite),
  or (c) the user interrupts with Ctrl+C.
- **Web streaming:** An optional `message_sink` callback allows the web UI to receive
  real-time plain-text status updates over WebSocket.

### 2. Planner (`furrow-planner`)
- **Type:** Python class (`PlannerAgent`) + Kilo subagent persona
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** `Plan` model with `tasks[]` (each having `id`, `description`, `files`,
  `dependencies`, `status`, `result`) and a `rationale` string.

### 3. Worker (`furrow-worker`)
- **Type:** Python class (`WorkerAgent`) + Kilo subagent persona
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.

### 4. Tester (`furrow-tester`)
- **Type:** Python class (`TesterAgent`) + Kilo subagent persona
- **Role:** Runs the project's test suite, then uses an LLM to evaluate pass/fail.
  Auto-detects test runners: pytest, npm/yarn/pnpm test, cargo test, go test.
  Returns a `TestResult` model with `passed`, `summary`, and `failures`.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

### 6. CLI
- **Type:** Click-based CLI (`furrow/cli/main.py`)
- **Commands:**
  - `furrow start <goal>` — runs the development loop
  - `furrow start <goal> --max-cycles 5` — run at most 5 cycles
  - `furrow web` — starts the FastAPI web server with WebSocket streaming

## LLM Support
The `LLMClient` supports multiple providers via `FURROW_PROVIDER`:
- **Anthropic** (`Provider.ANTHROPIC`): Claude models via the Anthropic API
- **OpenAI** (`Provider.OPENAI`): GPT models via the OpenAI API
- **Ollama** (`Provider.OLLAMA`): Local models via the Ollama server (no API key needed)

All providers include automatic retry with exponential backoff (3 attempts) on
transient API errors using `tenacity`.

## Logging
Uses `structlog` (with stdlib `logging` fallback) for structured application logging.
Call `configure_logging()` once at startup (done automatically by the CLI).

## Web UI
The FastAPI server (`furrow/web/server.py`) provides:
- `GET /` — HTML form to enter a goal
- `WebSocket /ws` — real-time streaming of orchestrator progress messages

## State
`LoopState` model tracks `goal`, `cycles`, `tasks`, `history`, and `completed` flag
for potential cross-session persistence.

## Files
```
furrow/
  __init__.py    # Public exports
  config.py      # Settings, models, logging config
  llm.py         # Multi-provider LLM client with retry
  cli/main.py    # Click CLI (start, web)
  core/
    orchestrator.py  # Main development loop
  agents/
    planner.py   # Task planning agent
    worker.py    # Task execution agent
    tester.py    # Test execution & evaluation agent
    prompts.py   # System prompts
  web/
    server.py    # FastAPI server with WebSocket streaming
tests/
  test_core.py       # Config model & orchestrator state tests
  test_llm.py        # LLM client provider/routing tests
  test_orchestrator.py  # Message sink & web endpoint tests
.kilo/
  agent/
    furrow.md          # Orchestrator persona
    furrow-planner.md  # Planner subagent persona
    furrow-worker.md   # Worker subagent persona
    furrow-tester.md   # Tester subagent persona
  command/
    furrow.md          # /furrow slash command
kilo.json              # Project config (default agent = furrow, open permissions)
pyproject.toml         # Build config, ruff, mypy
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- Run `furrow start "<goal>"` from the CLI.
- Run `furrow web` to start the web server, then open `http://localhost:8000`.
- Mid-loop user input is incorporated into the next cycle's plan.
