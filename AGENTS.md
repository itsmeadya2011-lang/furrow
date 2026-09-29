# AGENTS.md

This file provides coding guidelines for AI agents working on the Furrow project.

## Project Overview

**Furrow** is an autonomous coding agent that runs an infinite parallel development
loop. Given a goal, it repeatedly plans tasks, executes them concurrently across
multiple agent roles, tests the results, and iterates until the goal is complete.

The project is built on Python 3.10+ and uses async/await for concurrency.

## Code Style Guidelines

- **Language:** Python 3.10+
- **Line length:** 100 characters
- **Formatter/Linter:** [Ruff](https://docs.astral.sh/ruff/)
  - Line length is set to 100 in `[tool.ruff]`.
  - The `E501` (line-too-long) lint rule is ignored because formatting is
    handled by the Ruff formatter.
  - Run `ruff format .` to format and `ruff check .` to lint.
- **Type checking:** [mypy](https://mypy.readthedocs.io/)
  - Use `from __future__ import annotations` at the top of new modules.
  - Prefer built-in generic syntax (`list[str]`, `X | None`) over
    `typing.List` / `typing.Optional` where practical.
  - Run `mypy .` to type-check.

## Testing Requirements

- The test suite uses **pytest** with **pytest-asyncio** (`asyncio_mode = "auto"`).
- Test files live in `tests/` and use the `test_*.py` naming convention.
- **Before committing, run the full test suite:**
  ```bash
  pytest
  ```
- Add tests for new functionality and ensure all existing tests continue to pass.

## Key Architecture Notes

Furrow's core is an agent loop orchestrated by the `Orchestrator` class. Each
cycle follows this sequence:

1. **Plan** — `PlannerAgent.plan(goal)` produces a `Plan` of `TaskModel` items,
   each with a description, target files, and dependencies.
2. **Execute** — `WorkerAgent` instances run the planned tasks concurrently via
   `asyncio.gather`, updating each task's status (`completed` / `failed`).
3. **Test** — `TesterAgent.run(goal, tasks)` validates the results and returns a
   `TestResult` (pass/fail with summary and failures).

If tests fail, the failed items become the new `goal` and the loop repeats. The
loop continues indefinitely unless `max_cycles` is set to a non-zero value in
`FURROW_MAX_CYCLES` (0 = unlimited). Parallelism is bounded by
`FURROW_MAX_PARALLEL_TASKS`.

Data models (`Provider`, `TaskModel`, `Plan`, `TestResult`, `Settings`) are
defined in `furrow/config.py` using Pydantic. Settings are loaded from
environment variables with the `FURROW_` prefix and an optional `.env` file.

## File Structure Overview

```
furrow/                # Main package
├── __init__.py
├── cli/               # CLI entry point (click-based)
│   └── main.py        # `furrow` console script -> main()
├── config.py          # Settings, models (TaskModel, Plan, TestResult, Provider)
├── llm.py             # LLMClient — unified client for Anthropic/OpenAI/Ollama
├── agents/            # Agent roles
│   ├── __init__.py
│   ├── prompts.py     # Prompt templates
│   ├── planner.py     # PlannerAgent — generates task plans
│   ├── worker.py      # WorkerAgent — executes individual tasks
│   └── tester.py      # TesterAgent — validates results
├── core/              # Orchestration logic
│   ├── __init__.py
│   └── orchestrator.py  # Orchestrator — the agent loop (plan→execute→test→repeat)
└── web/               # HTTP API and web UI
    ├── __init__.py
    └── server.py      # FastAPI app / uvicorn server

tests/                 # pytest test suite
    ├── __init__.py
    └── test_core.py

# Project files (root)
pyproject.toml         # Project metadata, deps, ruff & mypy config
.env.example           # Example environment configuration
AGENTS.md              # This file
README.md
```
