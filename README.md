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

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.

## Configuration

Furrow can be configured via environment variables (prefixed with `FURROW_`) or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `FURROW_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model for planning |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model for worker tasks |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model for testing |
| `FURROW_ANTHROPIC_API_KEY` | - | Anthropic API key |
| `FURROW_OPENAI_API_KEY` | - | OpenAI API key |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Max parallel worker tasks |
| `FURROW_MAX_CYCLES` | `0` | Max development cycles (0 = unlimited) |
| `FURROW_WORKSPACE` | `.` | Working directory |
| `FURROW_LOG_LEVEL` | `INFO` | Log level |
| `FURROW_TEST_COMMAND` | auto-detect | Custom test command (e.g., `pytest -x`) |

## Features

- **Infinite development loop**: Plans, executes, tests, and repeats until goal is complete
- **Parallel execution**: Runs up to 5 tasks concurrently (configurable)
- **Persistent state**: Saves progress to `.furrow_state.json` for resumability
- **Task dependencies**: Tasks can declare dependencies for proper ordering
- **Multiple LLM providers**: Supports Anthropic, OpenAI, and Ollama
- **Web UI**: Run `furrow web` for a browser-based interface
- **Structured logging**: Uses structlog for JSON logging
- **Retry logic**: Automatic retries with exponential backoff for LLM calls

## Architecture

- **Orchestrator**: Main loop coordinator
- **Planner**: Breaks goals into parallelizable tasks
- **Worker**: Executes tasks with file read/write/execute tools
- **Tester**: Runs tests and analyzes results
