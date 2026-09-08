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
  3. Execute via parallel workers (capped at `max_parallel_tasks`)
  4. Test via tester agent (with auto-fix on failure)
  5. Report and repeat
- **Configuration:**
  - `max_cycles` (FURROW_MAX_CYCLES, default 0 = infinite)
  - `max_parallel_tasks` (FURROW_MAX_PARALLEL_TASKS, default 5)
  - `workspace` (FURROW_WORKSPACE, default: current directory)
- **State:** Persists task state to `.furrow-state.json` in the workspace between sessions.

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.
- **Capabilities:** Retry on LLM failure, JSON extraction from markdown code blocks.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task using LLM tool calling.
- **Tools available:** read_file, write_file, list_files, execute_command
- **Constraint:** 1-3 tool call rounds per task.
- **Capabilities:** Tool-calling loop with Anthropic/OpenAI/Ollama support.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Fixes failures automatically.
- **Capabilities:**
  - Smart test runner discovery based on project files (pytest, npm, cargo, go, make)
  - Auto-spawns a fix worker on test failure
  - Re-runs tests after fix to verify

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
  config.py            # Settings, TaskModel, Plan, TestResult, Provider
  llm.py               # LLMClient (Anthropic, OpenAI, Ollama, tool calling)
  agents/              # Planner, Worker, Tester agents
  core/                # Orchestrator loop
  cli/                 # CLI entry point
  web/                 # Web UI (FastAPI + WebSocket)
```

## Configuration

Set these environment variables (with `FURROW_` prefix):

| Variable | Description | Default |
|----------|-------------|---------|
| `FURROW_PROVIDER` | LLM provider | `anthropic` |
| `FURROW_MODEL` | Default model | `claude-sonnet-4-20250514` |
| `FURROW_PLANNER_MODEL` | Planner model | `claude-3-5-haiku-20241022` |
| `FURROW_WORKER_MODEL` | Worker model | `claude-3-5-sonnet-20241022` |
| `FURROW_TESTER_MODEL` | Tester model | `claude-3-5-sonnet-20241022` |
| `FURROW_MAX_CYCLES` | Max cycles (0 = infinite) | `0` |
| `FURROW_MAX_PARALLEL_TASKS` | Max parallel workers | `5` |
| `FURROW_WORKSPACE` | Workspace directory | Current dir |
| `FURROW_LOG_LEVEL` | Log level | `INFO` |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `FURROW_OLLAMA_BASE_URL` | Ollama base URL | `http://localhost:11434` |

## Usage
```bash
# CLI
furrow start "Add JWT auth to the API"
furrow start "Add JWT auth" --model gpt-4o --max-cycles 10

# Web UI
furrow web

# Via Kilo
/furrow "Add JWT auth to the API"
```

## Implementation Details

### Tool Calling
Workers use the `complete_with_tools()` method on `LLMClient`, which implements
an agent loop: the LLM can call tools (read_file, write_file, list_files,
execute_command), and the loop continues until the LLM returns text without
tool calls or reaches the max iteration limit (default 20).

### Retry Logic
All LLM API calls are wrapped with `tenacity` retry logic using exponential
backoff (3 attempts, 1-10s backoff).

### Logging
Structured logging via `structlog` throughout the codebase. Configure with
`FURROW_LOG_LEVEL`.

### Test Runner Discovery
The tester auto-detects test runners based on project files:
- `pytest.ini` / `setup.py` → `pytest -q`
- `pyproject.toml` → `python -m pytest -q`
- `package.json` → `npm test --silent`
- `Cargo.toml` → `cargo test -q`
- `go.mod` → `go test ./...`
- `Makefile` → `make test`

## Future Work
- Desktop GUI wrapper for non-TUI interaction.
- Git integration for automatic commit/push per cycle.
- Multi-agent collaboration across multiple Furrow instances.
