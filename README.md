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

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment (or use Ollama with `FURROW_PROVIDER=ollama`).

### CLI Options

```bash
furrow start "Add JWT auth" \
  --model claude-3-5-sonnet-20241022 \
  --max-cycles 10 \
  --max-parallel 3 \
  --log-level DEBUG
```

| Option | Description |
|---|---|
| `--model` | Override the LLM model |
| `--max-cycles` | Limit the number of planning cycles (0 = unlimited) |
| `--max-parallel` | Limit concurrent tasks per cycle |
| `--log-level` | Set logging verbosity (DEBUG, INFO, WARNING, ERROR) |

## Configuration

All settings are configurable via environment variables (prefixed with `FURROW_`) or a `.env` file. See `.kilo/FURROW.md` for the full list.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check furrow tests
```
