# Furrow

Furrow is an autonomous coding agent that runs an infinite parallel development loop: it plans a goal, executes tasks in parallel, tests the results, and iterates until the goal is complete. It breaks down large requests into independent tasks, assigns them to worker agents, and uses a tester agent to verify the outcome of each cycle.

## Install

```bash
pip install furrow
```

For development, follow the setup instructions below.

## Development

```bash
git clone <repo-url>
cd furrow
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # create .env.example if it doesn't exist
```

Set your API keys in `.env` (see [Configuration](#configuration)). At least one LLM provider key must be set.

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow start "Add JWT auth to the API" --model claude-3-5-sonnet-20241022
furrow web
```

- `furrow start` — run the agent loop for a goal in the terminal.
- `furrow web` — start a web UI (FastAPI) on `http://localhost:8000` with a WebSocket-backed chat.

## CLI Commands

| Command | Description |
| --- | --- |
| `furrow start [GOAL] [--model MODEL]` | Start the autonomous agent loop. Prompts for a goal if not provided. |
| `furrow web` | Launch the web server (FastAPI + uvicorn). |

## Development Tools

Run tests:

```bash
pytest
```

Run the linter:

```bash
ruff check .
```

Run type checks:

```bash
mypy furrow
```

## Architecture

Furrow's agent loop is driven by four components in `furrow/core/` and `furrow/agents/`:

- **Orchestrator** — `furrow/core/orchestrator.py`: runs the infinite loop. Each cycle it invokes the Planner, executes tasks in parallel via `asyncio.gather`, then runs the Tester. If tests fail, the next cycle is redirected to fix the failures.
- **Planner** — `furrow/agents/planner.py`: asks the LLM to break a goal into 1–5 parallelizable, independent tasks and returns a structured `Plan`.
- **Worker** — `furrow/agents/worker.py`: implements a single task from the plan against the target codebase.
- **Tester** — `furrow/agents/tester.py`: runs the project's test suite (pytest, npm, cargo, go, etc.) and uses the LLM to judge whether tests pass and report failures.

These are coordinated through `furrow/llm.py` (`LLMClient`), a thin async wrapper over the configured provider's API. `furrow/web/server.py` exposes the same loop over a FastAPI WebSocket for the web UI.

## Configuration

Configuration is loaded from `FURROW_*` environment variables (or a `.env` file). The provider keys can also be set as `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` directly.

| Variable | Default | Description |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | LLM provider (`anthropic`, `openai`, or `ollama`). |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model used by the agent loop. |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the Planner agent. |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the Worker agent. |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the Tester agent. |
| `FURROW_ANTHROPIC_API_KEY` | — | Anthropic API key. |
| `FURROW_OPENAI_API_KEY` | — | OpenAI API key. |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (used when `provider=ollama`). |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Maximum number of tasks executed in parallel per cycle. |
| `FURROW_MAX_CYCLES` | `0` | Maximum number of cycles (`0` = unlimited). |
| `FURROW_WORKSPACE` | current directory | Workspace root the agent operates on. |
| `FURROW_LOG_LEVEL` | `INFO` | Logging level. |
