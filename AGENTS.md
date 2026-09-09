# AGENTS.md - Furrow Project Guide

This file documents the Furrow project for contributors and agent sessions.

## Project Overview

Furrow is an autonomous coding agent that runs an infinite parallel development loop. It takes a high-level goal, plans it into independent tasks, executes them concurrently, tests the result, and repeats until the goal is met or the user stops it.

**Current version**: 0.1.0  
**Python**: >= 3.10  
**License**: MIT

## Architecture

```
Goal → Planner → Tasks → Workers (parallel) → Tester → Report → Repeat
```

- **Orchestrator** (`furrow/core/orchestrator.py`): Runs the main loop. Manages cycle state, batches tasks to respect `max_parallel_tasks`, and decides when to stop based on `max_cycles` or goal completion.
- **Planner** (`furrow/agents/planner.py`): Calls the LLM to decompose the goal into 1-5 independent tasks with dependencies.
- **Worker** (`furrow/agents/worker.py`): Reads relevant files, implements one task, and returns a summary.
- **Tester** (`furrow/agents/tester.py`): Runs project test commands, captures output, and classifies pass/fail.
- **LLM Client** (`furrow/llm.py`): Async client for Anthropic, OpenAI, and Ollama with automatic retries and token tracking.
- **Config** (`furrow/config.py`): Pydantic Settings with `FURROW_` env prefix. Defines `Provider`, `Plan`, `TaskModel`, `TestResult`, and runtime options.

## Key Modules

| Module | Purpose |
|---|---|
| `furrow/core/orchestrator.py` | Main loop, state tracking, batching, retry wrappers |
| `furrow/agents/planner.py` | Goal decomposition into parallel tasks |
| `furrow/agents/worker.py` | Single-task implementation |
| `furrow/agents/tester.py` | Test execution and result classification |
| `furrow/agents/prompts.py` | System prompts for planner, worker, tester |
| `furrow/llm.py` | Multi-provider LLM client with tenacity retries |
| `furrow/config.py` | Settings, enums, data models |
| `furrow/cli/main.py` | Click commands: `start`, `web` |
| `furrow/web/server.py` | FastAPI + WebSocket UI |

## How to Contribute

1. Fork and clone the repo.
2. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Make your change.
5. Run tests: `pytest`
6. Run lint: `ruff check . && ruff format .`
7. Run type check: `mypy furrow`
8. Open a PR with a clear description.

## Coding Standards

- **Formatter**: `ruff format` with default style.
- **Linter**: `ruff check` with line-length = 100.
- **Type hints**: Required on all function signatures and public methods.
- **Target**: Python 3.10 (`py310` in ruff config).
- **Imports**: Use `from __future__ import annotations` in all modules.
- **Async**: Prefer `async`/`await` for I/O. Use `asyncio.gather` for parallel work.
- **Logging**: Use `structlog` with contextual keys.
- **Testing**: Use `pytest` + `pytest-asyncio`. Tests go in `tests/`.

## Testing Commands

```bash
pytest                          # Run tests
pytest -q                       # Quiet mode
pytest tests/test_core.py       # Run specific file
```

## Common Tasks

### Add a new environment variable

1. Add the field to `Settings` in `furrow/config.py`.
2. Use `settings.field_name` anywhere in the codebase.
3. Document it in `README.md`.

### Add a new LLM provider

1. Add a new enum value to `Provider` in `furrow/config.py`.
2. Add a new `_complete_<provider>` method in `LLMClient` in `furrow/llm.py`.
3. Add retry rules for the provider's exceptions in `_is_retryable`.
4. Update the CLI provider choices and Web UI dropdown.

### Modify agent behavior

1. Update the system prompt in `furrow/agents/prompts.py`.
2. Adjust the agent class in `furrow/agents/<agent>.py`.
3. Update the orchestrator in `furrow/core/orchestrator.py` if loop logic changes.
4. Update `.kilo/agent/furrow.md` to match.

### Run the web UI locally

```bash
furrow web
# or
uvicorn furrow.web.server:app --reload
```
