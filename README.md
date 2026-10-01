# Furrow

Autonomous coding agent with an infinite parallel development loop.

Breaks down a goal into parallelizable tasks, executes them concurrently, tests
the results, and repeats until the goal is complete.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# CLI - run in terminal
furrow start "Add JWT auth to the API"
furrow start "Build a REST API" --model "claude-3-5-sonnet-20241022"
furrow start --log-level DEBUG "Write unit tests for the auth module"

# Web UI - start the FastAPI server, then open http://localhost:8000
furrow web
furrow web --host 0.0.0.0 --port 8000
```

## Configuration

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment, or use a `.env`
file with `FURROW_` prefixed variables:

```bash
# .env
FURROW_PROVIDER=anthropic
FURROW_MODEL=claude-sonnet-4-20250514
FURROW_PLANNER_MODEL=claude-3-5-haiku-20241022
FURROW_WORKER_MODEL=claude-3-5-sonnet-20241022
FURROW_TESTER_MODEL=claude-3-5-sonnet-20241022
FURROW_MAX_CYCLES=0          # 0 = infinite
FURROW_MAX_PARALLEL_TASKS=5
FURROW_LOG_LEVEL=INFO
```

For local development with Ollama:

```bash
FURROW_PROVIDER=ollama
FURROW_MODEL=llama3.1
FURROW_OLLAMA_BASE_URL=http://localhost:11434
```

## Development

```bash
pytest            # run tests
ruff check .      # lint
```

## Architecture

See [`.kilo/FURROW.md`](.kilo/FURROW.md) for full architecture details.
