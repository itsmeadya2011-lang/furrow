# Furrow

Autonomous coding agent with an infinite parallel development loop. Furrow plans a goal into 1-5 parallel tasks, executes them with worker agents (honoring task dependencies and a concurrency limit), runs the test suite, and repeats until the goal is complete, tests pass, or a cycle limit is reached.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow start "Refactor the parser" --provider ollama --max-cycles 10
furrow web                      # streaming web UI at http://localhost:8000
furrow web --host 0.0.0.0 --port 8080
```

### Options

- `start [--model M] [--provider anthropic|openai|ollama] [--max-cycles N]`
- `web [--host H] [--port P]`

### Configuration

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment. All settings can also be configured via `FURROW_`-prefixed environment variables (e.g. `FURROW_PROVIDER=ollama`, `FURROW_MAX_PARALLEL_TASKS=3`, `FURROW_OLLAMA_BASE_URL=http://localhost:11434`) or a `.env` file.

## Development

```bash
pip install -e ".[dev]"
pytest
```