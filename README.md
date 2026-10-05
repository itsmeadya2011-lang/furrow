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

### Options

```bash
furrow start --provider ollama --model llama3 --max-cycles 10 --state-file .furrow-state.json
furrow web --host 0.0.0.0 --port 8000
```

| Option | Description |
|---|---|
| `--provider` | `anthropic` (default), `openai`, or `ollama` |
| `--model` | Override the LLM model |
| `--max-cycles` | Stop after N planning cycles (0 = unlimited, the default) |
| `--log-level` | `DEBUG`, `INFO` (default), `WARNING`, or `ERROR` |
| `--state-file` | Persist goal/cycle/task state to a JSON file across cycles |

All options can also be set via `FURROW_`-prefixed environment variables (e.g. `FURROW_PROVIDER=ollama`, `FURROW_MAX_CYCLES=5`).

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment, or run an Ollama server locally for the `ollama` provider.

The web UI (`furrow web`) streams live cycle/task/test events over a WebSocket at `/ws`; `GET /health` returns service status.
