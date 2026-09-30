# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects. It breaks a high-level goal into parallelizable tasks, executes them concurrently, tests the results, and repeats until the goal is complete or the session is stopped.

Furrow is implemented both as a Python package (`furrow/`) and as a set of Kilo-native agent and command definitions (`.kilo/agent/`, `.kilo/command/`). The Python package provides the standalone orchestration engine; the `.kilo/` definitions wire it into the Kilo runtime.

## Components

### 1. Orchestrator (`furrow/core/orchestrator.py`)
- **Type:** Primary agent (also usable as a subagent) and package entry point
- **Role:** Runs the infinite development loop. Receives a goal, plans, delegates to parallel workers, tests, reports, and repeats.
- **Entry:** `Orchestrator(goal).run()` (async) or via the `/furrow` slash command / `furrow start` CLI.
- **Cycle cap:** `max_cycles` is a `@property` that reads from `settings.max_cycles` (default `0` = infinite). It is overridable via `run(max_cycles=...)`, and the loop body enforces it as a hard cap, breaking after the configured number of cycles.
- **Concurrency:** `max_parallel_tasks` is enforced as an `asyncio.Semaphore`, limiting concurrent `WorkerAgent` execution per cycle.
- **State persistence:** Progress is persisted via `AgentState` / `CycleState` models (see `furrow/config.py`), written to `.furrow/state.json` by `save_state()` and restored by `load_state()`.
- **Progress streaming:** `status_callback` is accepted by `run()` and invoked with state dictionaries for real-time progress streaming (used by the web server).

### 2. Planner (`furrow/agents/planner.py`)
- **Type:** Agent
- **Role:** Breaks a high-level goal into 1-5 parallelizable, independent tasks.
- **Output:** `Plan` model (see `furrow/config.py`) with `tasks[]` (each `id`, `description`, `files`, `dependencies`) and a `rationale` string. Validates the LLM JSON response and retries once on malformed output.

### 3. Worker (`furrow/agents/worker.py`)
- **Type:** Agent
- **Role:** Executes a single assigned `TaskModel`. Minimal, targeted changes only — no scope creep.
- **Concurrency:** Workers for all tasks in a cycle are dispatched in parallel via `asyncio.gather`.

### 4. Tester (`furrow/agents/tester.py`)
- **Type:** Agent
- **Role:** Auto-detects and runs the project test runner (pytest, npm/pnpm/yarn, cargo, go), then asks the LLM to classify pass/fail. Returns a `TestResult` (`passed`, `summary`, `failures[]`). On failure the orchestrator reformulates the goal as "Fix failing tests" for the next cycle.

### 5. LLMClient (`furrow/llm.py`)
- **Type:** Multi-provider async client
- **Role:** Unified completion + file I/O used by all agents. Lazily instantiates the selected provider's async client.

### 6. Configuration (`furrow/config.py`)
- **Type:** `pydantic-settings` `Settings`
- **Role:** Loads settings from environment (`FURROW_` prefix) and `.env`; provides typed models (`Plan`, `TaskModel`, `TestResult`) and the `Provider` enum.

### 7. CLI (`furrow/cli/main.py`)
- **Type:** Click CLI
- **Commands:** `furrow start <goal>` (runs the orchestrator loop), `furrow web` (launches the web server).
- **Entry point:** `furrow = "furrow.cli.main:main"` (see `pyproject.toml`).

### 8. Web Server (`furrow/web/server.py`)
- **Type:** FastAPI app served with Uvicorn
- **Role:** Browser UI to submit a goal; a WebSocket endpoint streams orchestrator output back to the client in real time.
- **HTTP endpoints:** `/health` returns server health and orchestrator running state; `/status` returns the current orchestrator state (goal, cycles, running flag).

## Loop

Each cycle performs five steps, repeating until the goal is complete or the user stops the session:

1. **Plan** — `PlannerAgent` breaks the goal into 1-5 parallel tasks.
2. **Execute** — `WorkerAgent` instances run concurrently in parallel (via `asyncio.gather`) for every task in the plan.
3. **Test** — `TesterAgent` runs the project test suite and classifies results.
4. **Report** — Results are printed; on test failure the goal is reformulated to target the failing tests.
5. **Repeat** — The next cycle begins with the updated goal/context.

The loop terminates when no tasks remain / the goal is deemed complete, or when the user interrupts (Ctrl+C). `max_cycles` (default `0` = infinite) caps the number of cycles when set.

## Configuration

Settings are defined in `furrow/config.py` as a `pydantic-settings` `Settings` model. Values load from the `FURROW_`-prefixed environment variables and a `.env` file.

| Setting | Default | Description |
|---|---|---|
| `provider` | `anthropic` | LLM provider (`anthropic`, `openai`, `ollama`) |
| `model` | `claude-sonnet-4-20250514` | Default model used for general completion |
| `planner_model` | `claude-3-5-haiku-20241022` | Model used by `PlannerAgent` |
| `worker_model` | `claude-3-5-sonnet-20241022` | Model used by `WorkerAgent` |
| `tester_model` | `claude-3-5-sonnet-20241022` | Model used by `TesterAgent` |
| `max_parallel_tasks` | `5` | Max tasks planned/executed per cycle |
| `max_cycles` | `0` | Cycle cap (`0` = infinite) |
| `test_timeout` | `120` | Max seconds to wait for test suite execution |
| `workspace` | current directory | Project root the agent operates in |
| `anthropic_api_key` | (env) | Anthropic API key |
| `openai_api_key` | (env) | OpenAI API key |
| `ollama_base_url` | `http://localhost:11434` | Ollama endpoint |
| `log_level` | `INFO` | Logging verbosity |

## LLMClient

`LLMClient` (`furrow/llm.py`) provides a single async interface over multiple providers:

- **Anthropic** (`AsyncAnthropic`) — default provider; uses `messages.create`.
- **OpenAI** (`AsyncOpenAI`) — uses `chat.completions.create`.
- Provider is selected by `settings.provider`. Clients are instantiated lazily on first use via the `anthropic` / `openai` properties and raise `ValueError` if the corresponding API key is unset.

Core methods:
- `complete(prompt, system="", model=None) -> str` — routes to the active provider.
- `read_file(path) -> str` and `write_file(path, content)` — async file I/O via `aiofiles`, creating parent directories as needed.
- `list_files(directory) -> list[str]` — returns files relative to a directory.

> `Provider.OLLAMA` is defined in `config.py` but is not yet wired into `complete()` (it raises `ValueError` for unsupported providers); only `anthropic` and `openai` are currently implemented.

## Files

```
.kilo/
  agent/
    furrow.md              # Orchestrator (primary agent) definition
    furrow-planner.md      # Planner subagent definition
    furrow-worker.md       # Worker subagent definition
    furrow-tester.md       # Tester subagent definition
  command/
    furrow.md              # /furrow slash command
furrow/
  __init__.py              # Package exports: LLMClient, Settings
  config.py                # Settings, Provider enum, Plan/TaskModel/TestResult models
  llm.py                   # LLMClient - multi-provider (Anthropic/OpenAI) completion + file I/O
  core/
    __init__.py            # Exports Orchestrator
    orchestrator.py        # Orchestrator - infinite loop: plan/execute/test/report/repeat
  agents/
    __init__.py            # Exports PlannerAgent, WorkerAgent, TesterAgent
    planner.py             # PlannerAgent - breaks goals into parallel tasks
    worker.py              # WorkerAgent - executes a single task
    tester.py              # TesterAgent - runs test suites, parses results
    prompts.py             # PLANNER_PROMPT, WORKER_PROMPT, TESTER_PROMPT
  cli/
    __init__.py            # Exports main
    main.py                # Click CLI: `furrow start`, `furrow web`
  web/
    __init__.py            # Exports app, run
    server.py              # FastAPI app + WebSocket endpoint + uvicorn runner
tests/
  __init__.py
  test_core.py             # Tests for Plan/TaskModel/TestResult parsing
kilo.json                  # Default agent = furrow, open permissions
pyproject.toml             # Package config, scripts entry point, dev deps
```

## Permissions

Open by default in `kilo.json`: `bash: allow`, `edit: allow`, `read: allow`.

## Usage

- Run `/furrow <goal>` in Kilo to start the loop.
- Or use the CLI: `furrow start "<goal>"` or `furrow web` to launch the browser UI.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## Testing

The project uses:
- **pytest** — test runner (`asyncio_mode = auto`, `testpaths = ["tests"]`). `tests/test_core.py` covers `Plan`, `TaskModel`, and `TestResult` parsing. The `test_timeout` setting (default `120`s) caps how long the `TesterAgent` waits for the detected test suite to complete before killing it.
- **ruff** — linter/formatter (`line-length = 100`, `target-version = "py310"`).
- **mypy** — static type checking.

Run with: `pytest`, `ruff check .`, `mypy furrow`.

## Future Work

Implemented:
- Standalone Python CLI wrapper (`furrow start` / `furrow web`) for non-TUI interaction.
- Web UI (FastAPI + WebSocket) for monitoring parallel agent activity.
- Multi-provider `LLMClient` supporting Anthropic and OpenAI.
- `pydantic-settings` configuration with environment and `.env` loading.
- Parallel worker execution via `asyncio.gather` and JSON-validated planning with retry.
- `max_cycles` enforcement in the loop body — reads from `settings.max_cycles` via the `self.max_cycles` property and caps cycles as a hard limit when set.
- `max_parallel_tasks` enforced as an `asyncio.Semaphore` concurrency limit for parallel workers.
- Persistent state file for goal/task tracking: `AgentState` / `CycleState` models written to `.furrow/state.json` via `save_state()` and restored via `load_state()`.

Remaining:
- `Provider.OLLAMA` support wired into `LLMClient.complete()`.
- Mid-loop user input incorporation into the next plan.
