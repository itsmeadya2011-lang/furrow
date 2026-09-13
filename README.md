# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Overview

Furrow orchestrates an infinite loop of **plan → execute → test** cycles using
parallel subagents. It breaks high-level goals into independent tasks, executes
them concurrently, runs tests, and repeats until the goal is complete.

## Architecture

| Component | Role |
|-----------|------|
| **Orchestrator** | Runs the infinite loop; plans, delegates, tests, repeats |
| **Planner** | Breaks goals into 1–5 parallelizable tasks |
| **Worker** | Executes a single task (code changes, file I/O) |
| **Tester** | Runs tests; diagnoses and fixes failures |
| **Web UI** | Real-time monitoring via WebSocket |

## Install

```bash
pip install -e .
```

## Usage

### CLI

```bash
furrow start "Add JWT auth to the API"
furrow start --model gpt-4
furrow web
```

### Programmatic

```python
import asyncio
from furrow.core.orchestrator import Orchestrator
from furrow.core.loop import run_loop, LoopResult

async def main() -> None:
    result: LoopResult = await run_loop(
        goal="Build a REST API",
        orchestrator=Orchestrator(goal="Build a REST API"),
        max_cycles=10,
    )
    print(result)

asyncio.run(main())
```

### Web UI

```bash
furrow web
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Configuration

Set environment variables (prefixed with `FURROW_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `FURROW_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model |
| `FURROW_MAX_CYCLES` | `0` (unlimited) | Max planning cycles before halting |
| `FURROW_WORKSPACE` | `.` | Working directory for file operations |
| `ANTHROPIC_API_KEY` | — | API key for Anthropic |
| `OPENAI_API_KEY` | — | API key for OpenAI |

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check
mypy furrow
```

## License

MIT
