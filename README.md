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

`furrow start` runs the development loop on a goal. `furrow web` opens the live
monitoring UI on http://localhost:8000 (events stream over the `/ws` websocket,
with a `/health` endpoint).

The loop runs across cycles and respects `FURROW_MAX_CYCLES` (0 = infinite).
Worker edits are applied to disk under `FURROW_WORKSPACE`, and state is persisted
to `.furrow/state.json` (gitignored).

### Environment

| Variable               | Description                                  |
|------------------------|----------------------------------------------|
| `ANTHROPIC_API_KEY`    | Anthropic API key (used when `FURROW_PROVIDER=anthropic`). |
| `OPENAI_API_KEY`       | OpenAI API key (used when `FURROW_PROVIDER=openai` or `ollama`). |
| `FURROW_PROVIDER`      | `anthropic` \| `openai` \| `ollama` (default: `anthropic`). |
| `FURROW_MODEL`         | Model name to use.                           |
| `FURROW_MAX_CYCLES`    | Max number of cycles (0 = infinite).         |
| `FURROW_WORKSPACE`     | Directory where worker edits are applied.   |
| `FURROW_OLLAMA_BASE_URL`| Base URL for an Ollama server (OpenAI-compatible `/v1/chat/completions`). |

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check furrow
mypy furrow
```
