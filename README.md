# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
# Start a new task
furrow start "Add JWT auth to the API"

# Resume from saved state
furrow start "Add JWT auth to the API" --resume

# Start web UI
furrow web
```

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.

## Features

- **Infinite development loop** - Plans, executes, tests, and repeats until goal is complete
- **Parallel execution** - Runs up to 5 tasks concurrently by default
- **Persistent state** - Saves progress to `.furrow_state.json`, resume with `--resume`
- **Real file operations** - Workers read/write files and run commands
- **Auto-fix tests** - Tester attempts to fix failing tests automatically
- **Multiple LLM providers** - Anthropic, OpenAI, and Ollama support
- **Structured logging** - JSON logs with structlog
- **Web UI** - Real-time monitoring at `http://localhost:8000`

## Configuration

Environment variables (prefix `FURROW_`):
- `FURROW_PROVIDER`: `anthropic` | `openai` | `ollama` (default: anthropic)
- `FURROW_MODEL`: Model name (default: claude-sonnet-4-20250514)
- `FURROW_ANTHROPIC_API_KEY` / `FURROW_OPENAI_API_KEY`: API keys
- `FURROW_OLLAMA_BASE_URL`: Ollama endpoint (default: http://localhost:11434)
- `FURROW_MAX_PARALLEL_TASKS`: Max concurrent workers (default: 5)
- `FURROW_MAX_CYCLES`: Max cycles before stopping (0 = unlimited)
- `FURROW_WORKSPACE`: Working directory
- `FURROW_LOG_LEVEL`: DEBUG | INFO | WARNING | ERROR
- `FURROW_STATE_FILE`: State file path (default: .furrow_state.json)
