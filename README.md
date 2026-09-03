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
furrow status
furrow stop
furrow version
```

## Environment Variables

Copy `.env.example` to `.env` and set the required values for your provider:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `FURROW_PROVIDER` | `anthropic` | LLM provider (`anthropic`, `openai`, or `ollama`) |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model for general completions |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the planner agent |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the worker agent |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the tester agent |
| `FURROW_ANTHROPIC_API_KEY` | | API key for Anthropic (required if provider is `anthropic`) |
| `FURROW_OPENAI_API_KEY` | | API key for OpenAI (required if provider is `openai`) |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL for Ollama server |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Maximum number of parallel tasks |
| `FURROW_MAX_CYCLES` | `0` | Maximum development cycles (`0` for unlimited) |
| `FURROW_WORKSPACE` | | Workspace directory for Furrow operations |
| `FURROW_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment if not using a `.env` file.
