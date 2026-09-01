# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Overview

Furrow is an autonomous coding agent that drives software development through a
plan → parallel worker → test loop. Given a high-level goal, Furrow uses a
**Planner** agent to break the goal into a sequence of actionable tasks. Those
tasks are then dispatched to **Workers** that execute in parallel — each agent
operates independently against the codebase. A **Tester** agent reviews the
changes against the goal and tests, and the result feeds back into the Planner,
which re-evaluates progress and emits the next cycle of work. This loop repeats
until the goal is satisfied or a cycle limit is reached.

### High-Level Workflow

1. You provide a goal (via CLI argument or interactive prompt).
2. The Planner analyzes the workspace and decomposes the goal into tasks.
3. Workers (up to `FURROW_MAX_PARALLEL_TASKS`) execute tasks in parallel.
4. The Tester verifies results and reports back to the Planner.
5. The loop repeats until the goal is done or `FURROW_MAX_CYCLES` is reached.

## Installation

Install from PyPI once published:

```bash
pip install furrow
```

For development (installs editable with dev dependencies):

```bash
pip install -e .[dev]
```

### Prerequisites

- Python 3.10+
- An API key for your chosen LLM provider (see [Configuration](#configuration)).

## Quick Start

Run Furrow with a goal on the command line:

```bash
furrow start "Add JWT authentication to the API"
```

Run Furrow interactively (it will prompt for your goal):

```bash
furrow start
```

Launch the web dashboard:

```bash
furrow web
```

Override the model for a single run with the `--model` flag:

```bash
furrow start "Refactor the auth module" --model claude-3-5-sonnet-20241022
furrow start "Write unit tests" --model gpt-4o
```

Set your API key in the environment before running:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
furrow start "Fix the login bug"
```

## Configuration

Furrow is configured via environment variables. All variables are optional
unless noted otherwise.

| Variable | Default | Description |
|----------|---------|-------------|
| `FURROW_PROVIDER` | `anthropic` | LLM provider (`anthropic`, `openai`, `ollama`) |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model used when no model is specified |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the Planner agent |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by Worker agents |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the Tester agent |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Maximum number of parallel worker agents |
| `FURROW_MAX_CYCLES` | `0` | Maximum planning cycles (0 = unlimited) |
| `FURROW_WORKSPACE` | `.` | Working directory for the agent |
| `FURROW_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (required when `FURROW_PROVIDER=anthropic`) |
| `OPENAI_API_KEY` | — | OpenAI API key (required when `FURROW_PROVIDER=openai`) |

Example: run with OpenAI and cap the loop at 10 cycles:

```bash
export FURROW_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export FURROW_MAX_CYCLES=10
furrow start "Add a /health endpoint"
```

## Architecture

Furrow follows a plan → parallel worker → test loop architecture. The Planner
decides what to do next, Workers execute tasks concurrently, and the Tester
validates the outcome. The Tester feeds its findings back to the Planner so the
loop can continue until the goal is complete.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Planner   │────▶│  Workers    │────▶│   Tester    │
│   Agent     │     │  (parallel) │     │   Agent     │
└─────────────┘     └─────────────┘     └─────────────┘
       ▲                                       │
       └───────────────────────────────────────┘
                    (loop until done)
```

Key components:

- **Planner**: decomposes the goal into discrete, actionable tasks.
- **Workers**: execute tasks in parallel against the codebase, respecting the
  `FURROW_MAX_PARALLEL_TASKS` limit.
- **Tester**: runs/validates tests and reports results back to the Planner.

## Development

### Setup

```bash
git clone https://github.com/your-org/furrow.git
cd furrow
pip install -e .[dev]
```

### Running tests

```bash
pytest
```

### Linting

```bash
ruff check furrow tests
```

### Type checking

```bash
mypy furrow
```

### Common dev workflow

```bash
# run tests
pytest
# lint
ruff check furrow tests
# type check
mypy furrow
```

## Troubleshooting

### Missing API key errors

If you see `Missing API key` or authentication errors, ensure the correct
environment variable is set for your provider:

- `ANTHROPIC_API_KEY` for `anthropic`
- `OPENAI_API_KEY` for `openai`
- `FURROW_PROVIDER` is set to the provider you intend to use

```bash
echo $ANTHROPIC_API_KEY   # should print a non-empty key
echo $FURROW_PROVIDER     # should be anthropic, openai, or ollama
```

### JSON parse failures from the LLM

Models occasionally emit malformed JSON. If the agent hangs or fails during
planning, try reducing the temperature or switching to a model with stronger
JSON output. You can also set `FURROW_LOG_LEVEL=DEBUG` to inspect the raw
model responses:

```bash
export FURROW_LOG_LEVEL=DEBUG
furrow start "..."
```

### Infinite loops

If the agent appears stuck in a cycle, limit the number of planning cycles with
`FURROW_MAX_CYCLES`:

```bash
export FURROW_MAX_CYCLES=20
furrow start "..."
```

### Installation issues

- Make sure you are using Python 3.10+ (`python --version`).
- Reinstall in editable mode after cloning: `pip install -e .[dev]`.
- Confirm the package resolves: `python -c "import furrow; print(furrow.__file__)"`.

## License

_Not yet determined. This section is a placeholder for future use._

## Contributing

Contributions are welcome. This section is a placeholder; please open an issue
or pull request with your proposed changes.
