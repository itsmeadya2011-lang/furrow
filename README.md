# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# CLI — runs the infinite loop until the goal is complete or stopped
furrow start "Add JWT auth to the API"

# Web UI — serves a browser-based interface at http://localhost:8000
furrow web

# Or via Python module
python -m furrow start "Add JWT auth to the API"
```

### Configuration

Set API keys and provider options via environment variables (all prefixed with `FURROW_`):

```bash
export FURROW_PROVIDER=anthropic        # or "openai" or "ollama"
export FURROW_ANTHROPIC_API_KEY=sk-...
export FURROW_OPENAI_API_KEY=sk-...
export FURROW_OLLAMA_BASE_URL=http://localhost:11434
export FURROW_MODEL=claude-sonnet-4-20250514
export FURROW_LOG_LEVEL=INFO
export FURROW_MAX_CYCLES=0    # 0 = unlimited
export FURROW_MAX_PARALLEL_TASKS=5
```

A `.env` file is also supported via `python-dotenv`.

### Architecture

Furrow runs a continuous loop: **Plan → Execute (parallel workers) → Test → Report**.
- **Planner** — breaks goals into 1–5 parallelizable tasks
- **Worker** — executes tasks, writing files via `FILE:` blocks in LLM responses
- **Tester** — runs tests, lint, and type checks
- **Orchestrator** — coordinates the loop with retry logic, cycle limits, and concurrency control

See `.kilo/FURROW.md` for the full architecture.

## Development

```bash
# Run tests
pytest -q

# Lint + type-check
ruff check .
mypy furrow/
```
