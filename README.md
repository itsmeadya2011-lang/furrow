# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install -e .
```

## Usage

```bash
# Start the autonomous development loop with a goal
furrow start "Add JWT auth to the API"

# Start with a limited number of cycles
furrow start "Add JWT auth to the API" --max-cycles 10

# Use a specific LLM provider and model
furrow start "Add JWT auth to the API" --provider openai --model gpt-4

# Resume from a saved state file
furrow start "Continue previous work" --resume

# Start the web UI
furrow web --port 8000
```

### CLI Options

```
furrow start GOAL [OPTIONS]

Options:
  --model TEXT          Override LLM model
  --provider TEXT       LLM provider: anthropic, openai, ollama
  --planner-model TEXT  Override planner model
  --worker-model TEXT   Override worker model
  --tester-model TEXT   Override tester model
  --max-cycles INTEGER  Maximum number of cycles (0 = unlimited)
  --resume              Resume from saved state file
  --workspace TEXT      Workspace directory
  --log-level TEXT      Log level (DEBUG, INFO, WARNING, ERROR)

furrow web [OPTIONS]

Options:
  --host TEXT    Host to bind to (default: 0.0.0.0)
  --port INTEGER Port to bind to (default: 8000)
```

## Configuration

Furrow is configured via environment variables with the `FURROW_` prefix or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `FURROW_PROVIDER` | anthropic | LLM provider (anthropic, openai, ollama) |
| `FURROW_MODEL` | claude-sonnet-4-20250514 | Default LLM model |
| `FURROW_PLANNER_MODEL` | claude-3-5-haiku-20241022 | Planner agent model |
| `FURROW_WORKER_MODEL` | claude-3-5-sonnet-20241022 | Worker agent model |
| `FURROW_TESTER_MODEL` | claude-3-5-sonnet-20241022 | Tester agent model |
| `FURROW_ANTHROPIC_API_KEY` | - | Anthropic API key |
| `FURROW_OPENAI_API_KEY` | - | OpenAI API key |
| `FURROW_OLLAMA_BASE_URL` | http://localhost:11434 | Ollama server URL |
| `FURROW_OLLAMA_MODEL` | llama3.1 | Ollama model |
| `FURROW_MAX_PARALLEL_TASKS` | 5 | Max parallel tasks per cycle |
| `FURROW_MAX_CYCLES` | 0 | Max cycles (0 = unlimited) |
| `FURROW_STATE_FILE` | .furrow_state.json | State persistence file |
| `FURROW_LOG_LEVEL` | INFO | Log level |
| `FURROW_LOG_FORMAT` | json | Log format (json, console) |

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.

## Development

```bash
# Run tests
pytest

# Lint
ruff check .

# Format
ruff format .

# Type check
mypy .
```

## Architecture

Furrow runs an infinite development loop:

1. **Planner** — breaks the goal into 1-5 parallelizable tasks
2. **Worker** — executes tasks in parallel (respecting dependencies), reading/writing files
3. **Tester** — runs the test suite and evaluates results
4. **Orchestrator** — drives the loop, tracks state, respects cycle limits

See `.kilo/FURROW.md` for the full architecture documentation.
