# AGENTS.md — Kilo Agent Configuration

## What is Furrow

Furrow is an autonomous coding agent with an infinite parallel development loop. It takes a high-level goal, breaks it into tasks, executes them in parallel across multiple worker agents, tests the results, and repeats until the goal is complete or no progress is being made.

## Implementations

There are two ways to run Furrow:

1. **Kilo TUI plugin** (`.kilo/`) — Invokes Furrow as a Kilo agent via `/furrow <goal>` inside the Kilo TUI. The agent files under `.kilo/agent/` define the planner, worker, tester, and orchestrator personas.
2. **Standalone Python package** (`furrow/`) — A self-contained package with a CLI (`furrow start`, `furrow web`) and an HTTP API (FastAPI + Uvicorn).

## Usage

```bash
# CLI
furrow start "Add JWT auth to the API"
furrow web

# Kilo TUI
/furrow "Add JWT auth to the API"
```

## Architecture

```
Goal → Orchestrator → Planner → Workers (parallel via asyncio.gather) → Tester → repeat
```

- **Orchestrator** (`furrow/core/orchestrator.py`) — Drives the loop: plan → execute in parallel → test → fix → repeat.
- **Planner** (`furrow/agents/planner.py`) — Decomposes the goal into independent tasks.
- **Workers** (`furrow/agents/worker.py`) — Execute tasks concurrently; each reads/writes files as needed.
- **Tester** (`furrow/agents/tester.py`) — Runs lint, type-checking, and tests; reports failures back to the planner for the next cycle.

Configuration lives in `furrow/config.py` (Pydantic `Settings`). The LLM client (`furrow/llm.py`) supports Anthropic and OpenAI providers.

## Development Setup

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy -p furrow
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key (provider = `anthropic`) | — |
| `OPENAI_API_KEY` | OpenAI API key (provider = `openai`) | — |
| `FURROW_PROVIDER` | LLM provider | `anthropic` |
| `FURROW_MODEL` | Default model for planner/worker/tester | `claude-sonnet-4-…` |
| `FURROW_PLANNER_MODEL` | Model used by the Planner | `claude-3-5-haiku-…` |
| `FURROW_WORKER_MODEL` | Model used by Workers | `claude-3-5-sonnet-…` |
| `FURROW_TESTER_MODEL` | Model used by the Tester | `claude-3-5-sonnet-…` |
| `FURROW_MAX_PARALLEL_TASKS` | Max concurrent workers | `5` |
| `FURROW_MAX_CYCLES` | Max planning cycles (`0` = unlimited) | `0` |
| `FURROW_WORKSPACE` | Working directory | cwd |
| `FURROW_LINT_ENABLED` | Run lint between cycles | `true` |
| `FURROW_TYPE_CHECK_ENABLED` | Run mypy between cycles | `true` |
| `FURROW_TEST_TIMEOUT` | Test command timeout (seconds) | `120` |
| `FURROW_LOG_LEVEL` | Log verbosity | `INFO` |

All `FURROW_*` variables can also be set in a `.env` file at the project root.
