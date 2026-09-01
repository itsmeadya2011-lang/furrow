# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Run a goal in the terminal (Ctrl+C to stop)
furrow start "Add JWT auth to the API"

# Bound the loop (0 = infinite, default)
furrow start "Refactor the cache layer" --max-cycles 3

# Switch provider/model
furrow start "Add dark mode" --provider openai --model gpt-4o

# Web UI with live websocket streaming
furrow web
```

Set one of the following in your environment:

- `ANTHROPIC_API_KEY` (default provider)
- `OPENAI_API_KEY`
- `OLLAMA_BASE_URL` (when using `--provider ollama`)

## Configuration

All settings are read from environment variables prefixed with `FURROW_`
(or from a `.env` file). See `furrow/config.py` for the full list.
Notable knobs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | `anthropic`, `openai`, or `ollama` |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Concurrency cap per cycle |
| `FURROW_MAX_CYCLES` | `0` | Stop after N cycles (0 = infinite) |
| `FURROW_TEST_TIMEOUT_SECONDS` | `120` | Per-command timeout for tests/lint |

## Development

```bash
pytest -q
ruff check furrow tests
```
