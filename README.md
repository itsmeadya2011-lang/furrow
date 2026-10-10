# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Overview

Furrow runs a plan → execute → test → repeat loop until a goal is complete.
It exists in two complementary forms:

1. **Kilo agent system** (`.kilo/`) — Agent personas that run via the Kilo
   runtime using the Task tool for parallel subagents.
2. **Python package** (`furrow/`) — A standalone async implementation with a
   CLI (Click + Rich TUI) and a web server (FastAPI + WebSocket).

### Architecture (Python)

| Component   | File                        | Role                                              |
|-------------|-----------------------------|---------------------------------------------------|
| Orchestrator | `furrow/core/orchestrator.py` | Runs the infinite loop: plan → execute → test   |
| Planner     | `furrow/agents/planner.py`   | Breaks a goal into 1–5 parallel tasks (JSON)     |
| Worker      | `furrow/agents/worker.py`    | Implements one task via LLM tool-use (file I/O)  |
| Tester      | `furrow/agents/tester.py`    | Runs the project's test suite, judges pass/fail  |
| LLMClient   | `furrow/llm.py`              | Anthropic / OpenAI / Ollama API client w/ tools   |
| Settings    | `furrow/config.py`           | Pydantic settings (env prefix `FURROW_`)          |

Workers call `read_file`, `write_file`, and `list_files` tools exposed through
`LLMClient.complete_with_tools()`.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow web
```

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (or `OLLAMA` provider via
`FURROW_PROVIDER=ollama`) in your environment.

## Configuration

All settings are configurable via environment variables with the `FURROW_` prefix:

| Setting              | Env var              | Default                         |
|----------------------|----------------------|---------------------------------|
| `provider`           | `FURROW_PROVIDER`    | `anthropic`                     |
| `model`              | `FURROW_MODEL`       | `claude-sonnet-4-20250514`      |
| `planner_model`      | `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022`   |
| `worker_model`       | `FURROW_WORKER_MODEL`  | `claude-3-5-sonnet-20241022`|
| `tester_model`       | `FURROW_TESTER_MODEL`  | `claude-3-5-sonnet-20241022`|
| `max_cycles`         | `FURROW_MAX_CYCLES`  | `0` (unlimited)                 |
| `max_parallel_tasks` | `FURROW_MAX_PARALLEL_TASKS` | `5`                       |
| `max_tool_calls`     | `FURROW_MAX_TOOL_CALLS` | `20`                        |
| `workspace`          | `FURROW_WORKSPACE`   | Current directory               |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
