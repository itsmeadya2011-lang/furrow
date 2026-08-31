# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Overview

Furrow breaks down coding goals into parallelizable tasks, executes them concurrently, runs tests, and iterates until the goal is complete. It uses LLM agents for planning, execution, and testing.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Orchestrator                            │
│  Runs the infinite loop: plan → execute → test → repeat     │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐
│Planner │  │ Worker 1 │  │ Worker N │  (parallel execution)
│ Agent  │  │  Agent   │  │  Agent   │
└────────┘  └──────────┘  └──────────┘
                  │
                  ▼
           ┌──────────┐
           │  Tester  │
           │  Agent   │
           └──────────┘
```

### Components

- **Orchestrator** - Main loop controller that manages cycles of planning, execution, and testing
- **PlannerAgent** - Breaks goals into 1-5 parallelizable tasks
- **WorkerAgent** - Executes individual tasks via LLM
- **TesterAgent** - Runs test suite and evaluates results

## Install

```bash
pip install furrow
```

## Quick Start

### CLI Usage

```bash
# Start with a goal
furrow start "Add JWT authentication to the API"

# Override the model
furrow start "Refactor the database layer" --model claude-3-opus-20240229

# Limit the number of cycles
furrow start "Fix all linting errors" --max-cycles 5

# Interactive mode (will prompt for goal)
furrow start
```

### Web Interface

```bash
# Start the web server
furrow web
```

Then open http://localhost:8000 in your browser.

## Configuration

Set environment variables or create a `.env` file:

```env
# Required (one of)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional
FURROW_PROVIDER=anthropic          # or "openai"
FURROW_MODEL=claude-sonnet-4-20250514
FURROW_PLANNER_MODEL=claude-3-5-haiku-20241022
FURROW_WORKER_MODEL=claude-3-5-sonnet-20241022
FURROW_TESTER_MODEL=claude-3-5-sonnet-20241022
FURROW_MAX_PARALLEL_TASKS=5
FURROW_MAX_CYCLES=0                # 0 = unlimited
FURROW_LLM_TIMEOUT=120             # seconds
FURROW_LLM_MAX_RETRIES=3
FURROW_LOG_LEVEL=INFO
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=furrow --cov-report=term-missing

# Lint
ruff check furrow/

# Type check
mypy furrow/
```

## API Reference

### Orchestrator

```python
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

client = LLMClient()
orch = Orchestrator(goal="Your goal", client=client, max_cycles=10)
await orch.run()
```

### LLMClient

```python
from furrow.llm import LLMClient
from furrow.config import Settings, Settings

settings = Settings(provider=Provider.ANTHROPIC, model="claude-sonnet-4-20250514")
client = LLMClient(settings=settings)
response = await client.complete("Your prompt", system="System prompt")
```

## Error Handling

Furrow includes automatic retry logic for transient LLM API errors:
- Rate limit errors are retried with exponential backoff
- Connection errors raise `LLMError` with descriptive message
- Max 3 retries by default (configurable via `FURROW_LLM_MAX_RETRIES`)

## License

MIT
