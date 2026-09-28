# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects.
It combines a Python SDK (installable package) with Kilo-native agent definitions.

## Architecture

Furrow has two complementary interfaces:

### 1. Python SDK (`furrow` package)
A standalone Python package that can be installed via pip and run via CLI or web UI.

```
furrow/
  __init__.py           # Public API exports
  config.py             # Pydantic models: Settings, Plan, TaskModel, TestResult
  llm.py                # LLMClient: multi-provider LLM client with tool-use loop
  cli/main.py           # Click CLI: furrow start, furrow web
  core/
    orchestrator.py     # Orchestrator: infinite plan→execute→test loop
    state.py            # AgentState: persistent JSON state across sessions
  agents/
    planner.py          # PlannerAgent: breaks goals into parallelizable tasks
    worker.py           # WorkerAgent: executes tasks with LLM tool-use
    tester.py           # TesterAgent: runs test suites
    prompts.py          # LLM prompts for planner, worker, tester
  web/
    server.py           # FastAPI + WebSocket web UI
```

### 2. Kilo Agent Definitions (`.kilo/`)
Native Kilo agent definitions that can be used within the Kilo TUI.

```
.kilo/
  agent/
    furrow.md           # Orchestrator (primary agent, 50 steps)
    furrow-planner.md   # Planner subagent
    furrow-worker.md    # Worker subagent
    furrow-tester.md    # Tester subagent
  command/
    furrow.md           # /furrow slash command
  FURROW.md             # This file
kilo.json               # Project config (default agent = furrow, open permissions)
```

## Development Loop

The `Orchestrator` runs this cycle continuously:

1. **Plan** — `PlannerAgent` breaks the goal into 1-5 parallelizable tasks
2. **Execute** — `WorkerAgent` instances run in parallel (bounded by `max_parallel_tasks`)
3. **Test** — `TesterAgent` runs the project's test suite
4. **Report** — Results are displayed and persisted to `furrow-state.json`
5. **Repeat** — Next cycle adapts based on test results

## LLM Client Features

- **Multi-provider**: Anthropic, OpenAI, and Ollama support
- **Retry with backoff**: Transient failures are retried up to 3 times
- **Structured logging**: Uses `structlog` for debuggable traces
- **Tool-use loop**: Workers receive file read/write, command execution, glob, and grep tools

## Configuration

Settings are loaded from environment variables with the `FURROW_` prefix:

| Setting              | Env Var               | Default                    |
|----------------------|-----------------------|----------------------------|
| `provider`           | `FURROW_PROVIDER`     | `anthropic`                |
| `model`              | `FURROW_MODEL`        | `claude-sonnet-4-20250514` |
| `planner_model`      | `FURROW_PLANNER_MODEL`| `claude-3-5-haiku-20241022`|
| `worker_model`       | `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022`|
| `tester_model`       | `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022`|
| `anthropic_api_key`  | `FURROW_ANTHROPIC_API_KEY` | None                  |
| `openai_api_key`     | `FURROW_OPENAI_API_KEY` | None                     |
| `ollama_base_url`    | `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434`|
| `max_parallel_tasks` | `FURROW_MAX_PARALLEL_TASKS` | `5`                |
| `max_cycles`         | `FURROW_MAX_CYCLES`   | `0` (unlimited)            |
| `workspace`          | `FURROW_WORKSPACE`    | Current directory          |
| `log_level`          | `FURROW_LOG_LEVEL`    | `INFO`                     |

## State Persistence

After each cycle, the orchestrator saves state to `furrow-state.json` in the
workspace directory. This includes:
- The original goal
- All plan/task history (per cycle)
- Completed and failed task IDs
- Running status

On startup, `AgentState.load()` reads this file to resume from where it left off.

## Usage

### CLI
```bash
furrow start "Add JWT auth to the API"
furrow start "Add JWT auth to the API" --max-cycles 10 --max-parallel 3
furrow web --host 0.0.0.0 --port 8000
```

### Web UI
Start the web server and open `http://localhost:8000` in a browser:
```bash
furrow web
```

### Kilo Integration
Run `/furrow <goal>` in the Kilo TUI to start the loop.
