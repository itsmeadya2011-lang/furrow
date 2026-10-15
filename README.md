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

Configuration via `.env` file (see `.env.example`). All settings use the `FURROW_` prefix.

### Providers

- **Anthropic** (default) — set `ANTHROPIC_API_KEY`
- **OpenAI** — set `FURROW_PROVIDER=openai` and `OPENAI_API_KEY`
- **Ollama** — set `FURROW_PROVIDER=ollama` and `FURROW_OLLAMA_BASE_URL`

### Settings

| Setting | Default | Description |
|---|---|---|
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default LLM model |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model for planning |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model for workers |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model for testing |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Max concurrent worker tasks |
| `FURROW_MAX_CYCLES` | `0` | Max cycles (0 = unlimited) |
