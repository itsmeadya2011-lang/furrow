# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

Or from source:

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Run interactively in the terminal
furrow start "Add JWT auth to the API"

# Or launch the web UI
furrow web

# Or run as a module
python -m furrow start "Add JWT auth to the API"
```

### Configuration

Set the following environment variables (or create a `.env` file from `.env.example`):

| Variable | Description | Default |
|---|---|---|
| `FURROW_PROVIDER` | LLM provider: `anthropic`, `openai`, or `ollama` | `anthropic` |
| `FURROW_MODEL` | Default model | `claude-sonnet-4-20250514` |
| `FURROW_PLANNER_MODEL` | Planner model | `claude-3-5-haiku-20241022` |
| `FURROW_WORKER_MODEL` | Worker model | `claude-3-5-sonnet-20241022` |
| `FURROW_TESTER_MODEL` | Tester model | `claude-3-5-sonnet-20241022` |
| `FURROW_ANTHROPIC_API_KEY` | Anthropic API key | — |
| `FURROW_OPENAI_API_KEY` | OpenAI API key | — |
| `FURROW_OLLAMA_BASE_URL` | Ollama base URL | `http://localhost:11434` |
| `FURROW_MAX_PARALLEL_TASKS` | Parallel workers per cycle | `5` |
| `FURROW_MAX_CYCLES` | Max cycles (0 = unlimited) | `0` |
| `FURROW_WORKSPACE` | Working directory | `.` |
| `FURROW_LOG_LEVEL` | Log level | `INFO` |

## Architecture

Furrow runs an infinite plan-execute-test loop:

1. **Plan** — The planner breaks the goal into 1-5 parallelizable tasks
2. **Execute** — Worker agents run in parallel, making file edits and writing them to disk
3. **Test** — The tester runs the test suite, linter, and type checker
4. **Repeat** — Failures are refocused into the next cycle's plan

The orchestrator saves state to `.furrow_state.json` after each cycle, allowing recovery across sessions.

## Programmatic API

```python
import asyncio
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

async def main():
    client = LLMClient()
    orch = Orchestrator(goal="Add dark mode toggle", client=client)

    async def reporter(event: str, payload: dict):
        print(f"[{event}]", payload)

    await orch.run(reporter=reporter)

asyncio.run(main())
```
