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

## Configuration

Furrow supports multiple LLM providers:

- **Anthropic** (default): Set `ANTHROPIC_API_KEY`
- **OpenAI**: Set `OPENAI_API_KEY`
- **Ollama** (local): Run Ollama locally and optionally set `OLLAMA_BASE_URL` (default `http://localhost:11434`)

You can configure Furrow via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `FURROW_PROVIDER` | `anthropic` | LLM provider (`anthropic`, `openai`, `ollama`) |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default LLM model |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model for planning |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model for workers |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model for testing |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Max concurrent worker tasks |
| `FURROW_MAX_CYCLES` | `0` | Max planning-execution cycles (0 = infinite) |
| `FURROW_WORKSPACE` | current directory | Workspace root |
| `FURROW_LOG_LEVEL` | `INFO` | Logging verbosity |

## Web UI

Run `furrow web` and open `http://localhost:8000` to monitor progress in real time via WebSocket streaming.
