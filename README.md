# Furrow

Autonomous coding agent with an infinite parallel development loop. It plans tasks, executes them in parallel, tests, and repeats until the goal is complete.

## Install

```bash
pip install furrow
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
# Start the development loop with a goal
furrow start "Add JWT auth to the API"

# Limit to a specific number of cycles
furrow start "Build a REST API" --cycles 10

# Override LLM models
furrow start "Add tests" --planner-model claude-3-5-haiku-20240620 --worker-model claude-3-5-sonnet-20240620

# Use a different provider
furrow start "Refactor auth" --provider openai --model gpt-4o

# Start the web UI
furrow web --port 8080
```

### Kilo TUI

```
/furrow "Add JWT auth to the API"
```

## Configuration

Settings can be configured via environment variables (prefix `FURROW_`) or a `.env` file:

| Setting | Default | Description |
|---------|---------|-------------|
| `provider` | `anthropic` | LLM provider: `anthropic`, `openai`, `ollama` |
| `model` | `claude-sonnet-4-20250514` | Default LLM model |
| `planner_model` | `claude-3-5-haiku-20241022` | Model for planning |
| `worker_model` | `claude-3-5-sonnet-20241022` | Model for task execution |
| `tester_model` | `claude-3-5-sonnet-20241022` | Model for testing |
| `max_parallel_tasks` | `5` | Max concurrent workers per cycle |
| `max_cycles` | `0` | Max cycles (0 = infinite) |
| `workspace` | `cwd` | Working directory |
| `log_level` | `INFO` | Logging level |

### Environment Variables

```bash
export FURROW_PROVIDER=openai
export FURROW_OPENAI_API_KEY=sk-...
export FURROW_PLANNER_MODEL=gpt-4o-mini
export FURROW_MAX_CYCLES=10
export FURROW_LOG_LEVEL=DEBUG
```

Required API keys:
- `ANTHROPIC_API_KEY` (or `FURROW_ANTHROPIC_API_KEY`) for Anthropic
- `OPENAI_API_KEY` (or `FURROW_OPENAI_API_KEY`) for OpenAI
- Ollama runs locally, no API key needed

## Features

- **Parallel task execution** with dependency resolution
- **Multi-provider LLM support** (Anthropic, OpenAI, Ollama)
- **Automatic retry** on LLM API failures (3 attempts, exponential backoff)
- **Structured logging** via structlog
- **Dependency-aware scheduling** — tasks run after their dependencies complete
- **Max cycles** limit for safety
- **Web UI** with live WebSocket streaming

## Testing

```bash
pytest tests/ -v
```

## License

MIT
