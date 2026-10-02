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

All settings are configurable via environment variables (prefixed `FURROW_`) or a `.env` file:

| Setting | Default | Description |
|---------|---------|-------------|
| `provider` | `anthropic` | LLM provider: `anthropic`, `openai`, or `ollama` |
| `model` | `claude-sonnet-4-20250514` | Default completion model |
| `planner_model` | `claude-3-5-haiku-20241022` | Model for planning |
| `worker_model` | `claude-3-5-sonnet-20241022` | Model for task execution |
| `tester_model` | `claude-3-5-sonnet-20241022` | Model for testing |
| `max_cycles` | `0` | Max cycles (0 = unlimited) |
| `max_parallel_tasks` | `5` | Concurrent worker tasks |
| `ollama_base_url` | `http://localhost:11434` | Ollama API URL |
| `state_file` | `.furrow-state.json` | Persistent state path |
| `log_level` | `INFO` | Logging verbosity |

## Advanced

- **Resume a session:** `furrow start "goal" --resume`
- **Limit cycles:** `furrow start "goal" --model claude-3-5-sonnet-20241022` (env: `FURROW_MAX_CYCLES=3`)
- **Web UI:** `furrow web` starts a FastAPI server with WebSocket streaming at `http://localhost:8000`
