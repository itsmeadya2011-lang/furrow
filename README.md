# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
# Run interactively (requires an LLM API key)
furrow start "Add JWT auth to the API"

# Start the web UI
furrow web

# Limit the number of auto-cycles
furrow start "Add JWT auth" --max-cycles 3

# Override concurrency and workspace
furrow start "Fix all bugs" --concurrency 3 --workspace /path/to/project
```

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment. Ollama is also
supported by setting `FURROW_PROVIDER=ollama` (ensure Ollama is running locally).

## CLI Options

| Option | Description |
|---|---|
| `--model` | Override the LLM model |
| `--max-cycles` | Maximum number of plan→execute→test cycles (0 = unlimited) |
| `--concurrency` | Maximum number of parallel worker tasks |
| `--workspace` | Working directory for code changes |

## Architecture

- **Orchestrator** — Runs the infinite loop: plan → execute (parallel) → test → repeat
- **Planner** — Breaks goals into 1-5 parallelizable tasks
- **Worker** — Executes tasks with LLM tool calling (read, write, edit, bash)
- **Tester** — Runs the test suite and evaluates results

See `.kilo/FURROW.md` for full architecture details.
