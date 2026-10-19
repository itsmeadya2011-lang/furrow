# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow web
```

Alternatively, run as a module:

```bash
python -m furrow start "Add JWT auth to the API"
```

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.

## Architecture

Furrow consists of an orchestrator loop that drives three agents: a planner
that decomposes a goal into tasks, a worker that executes tasks, and a tester
that validates results. Tasks run in parallel up to a configurable limit, and
the loop continues until the goal is complete or the maximum cycle count is
reached. A FastAPI web UI (`furrow web`) provides a browser-based interface for
monitoring and triggering runs.

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

Key variables include `FURROW_PROVIDER` (anthropic, openai, ollama),
`FURROW_MODEL` and per-agent model overrides, `ANTHROPIC_API_KEY` /
`OPENAI_API_KEY`, `FURROW_OLLAMA_BASE_URL`, `FURROW_MAX_PARALLEL_TASKS`,
`FURROW_MAX_CYCLES`, `FURROW_WORKSPACE`, and `FURROW_LOG_LEVEL`. See
`.env.example` for the full list and defaults.

## Development

Install with dev dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

Lint and typecheck:

```bash
ruff check
mypy .
```

## Web UI

Run `furrow web` to start the FastAPI server, then open the local URL printed to
the console to view runs, tasks, and agent output in the browser.
