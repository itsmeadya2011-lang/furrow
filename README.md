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

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FURROW_PROVIDER` | LLM provider: `anthropic`, `openai`, or `ollama` |
| `FURROW_MODEL` | Default model name |
| `FURROW_PLANNER_MODEL` | Model for planning agent |
| `FURROW_WORKER_MODEL` | Model for worker agent |
| `FURROW_TESTER_MODEL` | Model for tester agent |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `FURROW_OLLAMA_BASE_URL` | Ollama base URL (default: `http://localhost:11434`) |
| `FURROW_MAX_PARALLEL_TASKS` | Max parallel tasks (default: 5) |
| `FURROW_MAX_CYCLES` | Max development cycles (default: 0 = infinite) |

## Web UI

```bash
furrow web
```

Open `http://localhost:8000` to monitor the development loop in real-time.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Lint
ruff check furrow/

# Type check
mypy furrow/
```
