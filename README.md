# Furrow

Autonomous coding agent with an infinite parallel development loop.

Furrow orchestrates multiple AI coding agents — a **planner**, parallel **workers**,
and **testers** — in a continuous loop. The planner breaks a goal down into
sub-tasks, workers implement them concurrently, testers verify the results, and
the cycle repeats until the goal is complete.

## Features

- **Infinite development loop** — continuously plans, implements, and tests until objectives are met.
- **Parallel execution** — runs multiple worker agents concurrently for fast iteration.
- **Planner / Worker / Tester architecture** — specialized agents for strategic planning, implementation, and verification.
- **Multi-provider LLM support** — works with Anthropic, OpenAI, and Ollama.
- **Per-role model selection** — assign dedicated models to the planner, workers, and testers.
- **Web dashboard** — monitor running tasks and cycles through a built-in web UI.
- **Configurable** — tune parallelism, cycle limits, workspace, and logging via environment variables.

## Installation

```bash
pip install furrow
```

Or install from source:

```bash
git clone https://github.com/furrow-dev/furrow.git
cd furrow
pip install -e .
```

## Quick Start

Run an autonomous development task:

```bash
furrow start "Add JWT authentication to the REST API"
```

Launch the web dashboard:

```bash
furrow web
```

Open `http://localhost:8000` in your browser to monitor progress.

Set at least one LLM provider API key:

```bash
export ANTHROPIC_API_KEY="sk-..."
# or
export OPENAI_API_KEY="sk-..."
```

## Configuration

Furrow is configured via environment variables. The `Settings` class in
`furrow/config.py` uses the `FURROW_` prefix with `pydantic-settings`.

| Variable | Description | Default |
|---|---|---|
| `FURROW_PROVIDER` | LLM provider to use (`anthropic`, `openai`, `ollama`) | `anthropic` |
| `FURROW_MODEL` | Default model used when no role-specific model is set | `claude-sonnet-4-20250514` |
| `FURROW_PLANNER_MODEL` | Model used by the planner agent | `claude-3-5-haiku-20241022` |
| `FURROW_WORKER_MODEL` | Model used by worker agents | `claude-3-5-sonnet-20241022` |
| `FURROW_TESTER_MODEL` | Model used by the tester agent | `claude-3-5-sonnet-20241022` |
| `ANTHROPIC_API_KEY` | Anthropic API key (required when provider is `anthropic`) | — |
| `OPENAI_API_KEY` | OpenAI API key (required when provider is `openai`) | — |
| `FURROW_OLLAMA_BASE_URL` | Base URL for the Ollama API (when provider is `ollama`) | `http://localhost:11434` |
| `FURROW_MAX_PARALLEL_TASKS` | Maximum number of worker tasks run in parallel | `5` |
| `FURROW_MAX_CYCLES` | Maximum development cycles before stopping (`0` = unlimited) | `0` |
| `FURROW_WORKSPACE` | Path to the workspace directory where code is written | Current directory |
| `FURROW_LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |

### Example: Ollama setup

```bash
export FURROW_PROVIDER=ollama
export FURROW_MODEL=llama3.1:8b
export FURROW_PLANNER_MODEL=llama3.1:70b
export FURROW_WORKER_MODEL=llama3.1:8b
export FURROW_TESTER_MODEL=llama3.1:70b
export FURROW_OLLAMA_BASE_URL=http://localhost:11434
export FURROW_MAX_PARALLEL_TASKS=8
```

## Commands

```text
furrow start "<task description>"  Run an autonomous development task.
furrow web                        Start the web dashboard server.
furrow --help                     Show available commands and options.
```

### `furrow start`

```bash
furrow start "Refactor the user service and add unit tests"
```

Options:

- `--model` — override the default model for this run.

### `furrow web`

```bash
furrow web
```

Starts a FastAPI server (via Uvicorn) on `0.0.0.0:8000` with:

- `GET /` — web dashboard (serves an HTML page with a WebSocket console).
- `WebSocket /ws` — receives JSON with a `goal` field and streams plan/test progress back.

## Architecture

Furrow follows a **Planner / Worker / Tester / Orchestrator** architecture:

1. **Planner** (`furrow.agents.planner.PlannerAgent`) — Receives the high-level goal and decomposes it into 1–5 concrete, parallelizable sub-tasks. Each sub-task is a `TaskModel` with an `id`, `description`, `files`, and `dependencies`. Returns a `Plan` object.

2. **Orchestrator** (`furrow.core.orchestrator.Orchestrator`) — The central coordinator. It runs a cycle loop:
   - Calls the planner to generate a plan.
   - Collects all non-completed tasks from the plan.
   - Dispatches tasks to workers concurrently using `asyncio.Semaphore(max_parallel_tasks)`.
   - Collects results from all workers.
   - Invokes the tester to verify completed work.
   - Repeats until there are no tasks left or `max_cycles` is reached (`0` = unlimited).

3. **Workers** (`furrow.agents.worker.WorkerAgent`) — Execute the individual sub-tasks in parallel. Each worker receives a `TaskModel`, generates code, and writes files to the workspace via `LLMClient.write_file`.

4. **Tester** (`furrow.agents.tester.TesterAgent`) — Runs after each cycle to verify that worker output meets objectives. Returns a `TestResult` with a `passed` flag, `summary`, and `failures` list. Test failures are fed back into the planner so the next cycle can address them.

This design enables Furrow to iteratively refine a codebase: plan → implement → test → replan, indefinitely or until a cycle budget is exhausted.

```
┌──────────────────────────────────────────────────────┐
│                    Orchestrator                        │
│                                                        │
│   ┌────────┐    ┌──────────┐    ┌─────────┐          │
│   │ Planner │──▶│  Workers │───▶│ Tester  │          │
│   └────────┘    └──────────┘    └─────────┘          │
│         ▲              │                 │            │
│         │              ▼                 │            │
│         └────────── Feedback ─────────────┘            │
└──────────────────────────────────────────────────────┘
```

### Components

| Component | Module | Description |
|---|---|---|
| Orchestrator | `furrow.core.orchestrator.Orchestrator` | Runs the plan → execute → test cycle loop. |
| Planner | `furrow.agents.planner.PlannerAgent` | Decomposes goals into task plans. |
| Worker | `furrow.agents.worker.WorkerAgent` | Implements individual tasks in parallel. |
| Tester | `furrow.agents.tester.TesterAgent` | Validates completed work after each cycle. |
| LLM Client | `furrow.llm.LLMClient` | Unified interface to Anthropic/OpenAI/Ollama APIs; also handles file read/write. |
| CLI | `furrow.cli.main` | Click-based CLI with `start` and `web` commands. |
| Web Server | `furrow.web.server` | FastAPI app serving a real-time dashboard over WebSocket. |

## License

Furrow is released under the [MIT License](LICENSE).
