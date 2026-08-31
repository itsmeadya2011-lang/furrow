# Furrow

Autonomous coding agent with an infinite parallel development loop. Furrow breaks down high-level goals into parallelizable tasks, executes them via worker subagents, tests the results, and iterates until completion.

## Features

- **Parallel Task Execution**: Automatically breaks goals into independent tasks and runs them concurrently
- **Dependency-Aware Scheduling**: Tasks can declare dependencies and will be executed in the correct order
- **Multi-Provider LLM Support**: Works with Anthropic (Claude) and OpenAI models
- **Web UI**: Browser-based interface with real-time streaming output
- **CLI Interface**: Run from the terminal with rich console output
- **Automatic Testing**: Runs your test suite and iterates on failures
- **Error Recovery**: Built-in retry logic for transient failures

## Install

```bash
pip install furrow
```

## Quick Start

### CLI Usage

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...

# Run with a goal
furrow start "Add JWT authentication to the API"

# Run with custom model
furrow start "Refactor the database layer" --model claude-3-5-sonnet-20241022

# Interactive mode (will prompt for goal)
furrow start
```

### Web UI

```bash
furrow web
```

Then open http://localhost:8000 in your browser.

## Configuration

Configuration is via environment variables (prefix with `FURROW_`) or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `FURROW_PROVIDER` | `anthropic` | LLM provider: `anthropic` or `openai` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model for all agents |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model for planning (faster/cheaper) |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model for task execution |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model for testing |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Maximum concurrent tasks |
| `FURROW_MAX_CYCLES` | `0` | Max cycles (0 = unlimited) |
| `FURROW_WORKSPACE` | Current directory | Working directory for file operations |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `OPENAI_API_KEY` | - | OpenAI API key |

### Example .env

```env
FURROW_PROVIDER=anthropic
FURROW_PLANNER_MODEL=claude-3-5-haiku-20241022
FURROW_WORKER_MODEL=claude-3-5-sonnet-20241022
FURROW_MAX_CYCLES=10
ANTHROPIC_API_KEY=sk-ant-...
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Orchestrator                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Planner  │→ │  Worker  │→ │  Worker  │→ │  Tester  │   │
│  │  Agent   │  │  Agent 1 │  │  Agent 2 │  │  Agent   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│       ↑                                          │          │
│       └──────────────────────────────────────────┘          │
│                    (iterate until done)                      │
└─────────────────────────────────────────────────────────────┘
```

### Agents

1. **PlannerAgent**: Breaks the goal into 1-5 parallelizable tasks with optional dependencies
2. **WorkerAgent**: Executes individual tasks by reading files, generating code changes, and writing results
3. **TesterAgent**: Runs the test suite and evaluates if the goal is complete

### Task Lifecycle

1. Planner creates a plan with tasks
2. Workers execute tasks (respecting dependencies)
3. Tester validates the results
4. If tests fail, the goal is updated and the cycle repeats

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Type check
mypy furrow
```

## License

MIT
