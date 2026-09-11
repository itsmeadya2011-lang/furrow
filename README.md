# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Features

- **Infinite Development Loop**: Runs continuously until the goal is complete
- **Parallel Task Execution**: Breaks goals into independent tasks that run concurrently
- **Automatic Testing**: Runs tests after each cycle and fixes failures
- **State Persistence**: Saves progress to resume after interruption
- **Web UI**: Monitor and control Furrow from a browser
- **Mid-loop Feedback**: Provide guidance during execution
- **Structured Logging**: JSON or human-readable logs with structlog

## Install

```bash
pip install furrow
```

## Usage

### CLI

```bash
# Start with a goal
furrow start "Add JWT auth to the API"

# Override model
furrow start "Add JWT auth" --model gpt-4

# Limit cycles and parallelism
furrow start "Build a REST API" --max-cycles 10 --max-parallel 3

# Enable debug logging
furrow start "Refactor code" --log-level DEBUG --log-json
```

### Web UI

```bash
furrow web
# Open http://localhost:8000 in your browser
```

The web UI allows you to:
- Start new goals
- Send feedback mid-execution
- Pause/resume the orchestrator
- View real-time output

### Environment Variables

```bash
# Required: API key for your provider
export ANTHROPIC_API_KEY=your_key_here
# or
export OPENAI_API_KEY=your_key_here

# Optional configuration
export FURROW_PROVIDER=anthropic          # anthropic, openai, ollama
export FURROW_MODEL=claude-sonnet-4-20250514
export FURROW_PLANNER_MODEL=claude-3-5-haiku-20241022
export FURROW_WORKER_MODEL=claude-3-5-sonnet-20241022
export FURROW_TESTER_MODEL=claude-3-5-sonnet-20241022
export FURROW_MAX_PARALLEL_TASKS=5
export FURROW_MAX_CYCLES=0                # 0 = unlimited
export FURROW_LOG_LEVEL=INFO
export FURROW_LOG_JSON=false
```

## Architecture

Furrow consists of four agents working in a loop:

1. **Orchestrator** (`furrow`) - Runs the infinite loop, coordinates all agents
2. **Planner** (`furrow-planner`) - Breaks goals into parallelizable tasks
3. **Worker** (`furrow-worker`) - Executes individual tasks with file operations
4. **Tester** (`furrow-tester`) - Runs tests, fixes failures

Each cycle:
1. **Plan**: Planner creates 1-5 independent tasks based on the goal
2. **Execute**: Workers run tasks in parallel (respecting dependencies)
3. **Test**: Tester runs test suite and fixes any failures
4. **Repeat**: Loop continues until goal is complete

## State Persistence

Furrow automatically saves state to `.furrow_state.json` in the workspace:
- Current goal and cycle count
- All planned tasks and their status
- Completed task IDs

Run `furrow start` again in the same directory to resume from where you left off.

## Python API

```python
import asyncio
from furrow import Orchestrator

async def main():
    orchestrator = Orchestrator(goal="Build a REST API with FastAPI")
    await orchestrator.run()

asyncio.run(main())
```

## Configuration

Create a `.env` file or use environment variables with `FURROW_` prefix:

```env
FURROW_PROVIDER=anthropic
FURROW_MODEL=claude-sonnet-4-20250514
FURROW_MAX_PARALLEL_TASKS=5
FURROW_MAX_CYCLES=10
FURROW_LOG_LEVEL=DEBUG
```

## License

MIT