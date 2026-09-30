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

Furrow reads configuration from environment variables with the `FURROW_` prefix
(or a `.env` file). The defaults are:

| Setting            | Default                  | Env var             |
| ------------------ | ------------------------ | ------------------- |
| `provider`         | `anthropic`              | `FURROW_provider`   |
| `model`            | `claude-sonnet-4-20250514` | `FURROW_model`   |
| `max_parallel_tasks` | `5`                    | `FURROW_max_parallel_tasks` |
| `max_cycles`       | `0` (unlimited)          | `FURROW_max_cycles` |
| `ollama_base_url`  | `http://localhost:11434` | `FURROW_ollama_base_url` |
| `workspace`        | current directory        | `FURROW_workspace`  |

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment. For local
models via Ollama, set `FURROW_provider=ollama` and `OLLAMA_API_KEY` is not
required (Ollama ignores the key). To bound a run, set `FURROW_max_cycles=N`.

A small state file is written to `<workspace>/.furrow/state.json` after each
cycle so progress survives across sessions.
