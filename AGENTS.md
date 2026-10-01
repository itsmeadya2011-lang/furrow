# AGENTS.md - Project-wide agent instructions for Kilo

This file provides guidance to coding agents working on the Furrow project.

## Project Overview

Furrow is an autonomous coding agent that runs an infinite parallel development loop.
It breaks down goals into parallelizable tasks, executes them concurrently, tests
the results, and repeats until the goal is complete.

## Architecture

- `.kilo/` - Kilo-native multi-agent definitions (orchestrator + subagents)
- `furrow/` - Standalone Python package mirroring the same architecture
- `furrow/core/orchestrator.py` - Core loop: Plan -> Execute (parallel) -> Test -> Repeat
- `furrow/agents/` - Planner, Worker, Tester agent classes
- `furrow/cli/main.py` - Click CLI entry point
- `furrow/web/server.py` - FastAPI web UI with WebSocket support

## Development

```bash
pip install -e ".[dev]"
pytest           # run tests
ruff check .     # lint
```

## Environment

- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` required for LLM calls
- Settings use `FURROW_` env prefix (e.g., `FURROW_MAX_CYCLES`, `FURROW_MODEL`)
- `.env` file is auto-loaded

## Key Settings

- `FURROW_PROVIDER` - anthropic, openai, or ollama (default: anthropic)
- `FURROW_MODEL` - default model for non-role-specific calls
- `FURROW_PLANNER_MODEL` - model used for task planning
- `FURROW_WORKER_MODEL` - model used for task execution
- `FURROW_TESTER_MODEL` - model used for test evaluation
- `FURROW_MAX_CYCLES` - max loop iterations (0 = infinite)
- `FURROW_MAX_PARALLEL_TASKS` - concurrent worker limit (default: 5)
