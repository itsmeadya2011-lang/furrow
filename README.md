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

## Features

- **Multi-cycle development loop**: Plans, executes, tests, and iterates until goals are met
- **Parallel task execution**: Runs multiple independent tasks simultaneously
- **Automatic retry**: Failed tasks are retried up to 3 times
- **Context-aware planning**: Each cycle builds on previous work
- **File operations**: Workers can read, write, and list files
- **Multiple LLM providers**: Supports Anthropic, OpenAI, and Ollama
- **Web UI**: Real-time progress streaming via WebSocket
- **Structured logging**: JSON-formatted logs for observability

## Configuration

Configure via environment variables (prefix `FURROW_`):

| Variable | Description | Default |
|----------|-------------|---------|
| `FURROW_PROVIDER` | LLM provider: `anthropic`, `openai`, `ollama` | `anthropic` |
| `FURROW_MODEL` | Default model | `claude-sonnet-4-20250514` |
| `FURROW_PLANNER_MODEL` | Model for planning | `claude-3-5-haiku-20241022` |
| `FURROW_WORKER_MODEL` | Model for code execution | `claude-3-5-sonnet-20241022` |
| `FURROW_TESTER_MODEL` | Model for test evaluation | `claude-3-5-sonnet-20241022` |
| `FURROW_ANTHROPIC_API_KEY` | Anthropic API key | - |
| `FURROW_OPENAI_API_KEY` | OpenAI API key | - |
| `FURROW_OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `FURROW_MAX_CYCLES` | Maximum development cycles (0 = unlimited) | `0` |
| `FURROW_MAX_PARALLEL_TASKS` | Max concurrent tasks | `5` |
| `FURROW_LOG_LEVEL` | Log level | `INFO` |

## Web UI

Start the web interface:

```bash
furrow web
```

Then open http://localhost:8000 in your browser. Enter a goal and watch the real-time progress as Furrow plans, executes, and tests.
