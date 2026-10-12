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

With options:

```bash
furrow start --provider ollama --model llama3 --max-cycles 10 "Migrate to SQLAlchemy 2.0"
furrow web --port 8080
```

## Configuration

Environment variables use the `FURROW_` prefix. A `.env` file is also read.

| Variable | Description |
|----------|-------------|
| `FURROW_PROVIDER` | Provider: `anthropic` (default), `openai`, or `ollama` |
| `FURROW_MODEL` | Default model for all agents |
| `FURROW_PLANNER_MODEL` | Model for the planner agent |
| `FURROW_WORKER_MODEL` | Model for worker agents |
| `FURROW_TESTER_MODEL` | Model for the tester agent |
| `FURROW_ANTHROPIC_API_KEY` | Anthropic API key |
| `FURROW_OPENAI_API_KEY` | OpenAI API key |
| `FURROW_OLLAMA_BASE_URL` | Ollama base URL (default: `http://localhost:11434/v1`) |
| `FURROW_MAX_CYCLES` | Maximum development cycles (default: 10) |
| `FURROW_MAX_PARALLEL_TASKS` | Max concurrent worker tasks (default: 4) |
| `FURROW_TEST_COMMAND` | Test command to run (default: `pytest -q`) |
| `FURROW_WEB_HOST` | Web UI bind host (default: `127.0.0.1`) |
| `FURROW_WEB_PORT` | Web UI port (default: `8000`) |

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.

## Providers

- **anthropic** (default) — Uses Anthropic models (Claude).
- **openai** — Uses OpenAI models (GPT-4o, etc.).
- **ollama** — Local models via Ollama (OpenAI-compatible endpoint, no API key required).

## How it works

The planner breaks the goal into parallel tasks. Workers execute tasks concurrently, bounded by `max_parallel_tasks`. The tester runs the project's test suite. The loop repeats until tests pass or `max_cycles` is reached. The web UI streams live progress over a websocket.