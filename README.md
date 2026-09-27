# Furrow

Autonomous coding agent with an infinite parallel development loop.

Furrow receives a high-level goal, breaks it into parallelizable tasks, executes those
tasks concurrently, runs tests, and repeats—fixing failures and progressing toward the goal
until it is complete.

## Installation

### From PyPI

```bash
pip install furrow
```

### From Source

```bash
git clone <repo-url>
cd furrow
pip install -e ".[dev]"
```

## Quick Start

```bash
# Copy the example environment file and fill in your keys
cp .env.example .env

# Start an autonomous development loop
furrow start "Add JWT authentication to the API"

# Or launch the web monitoring UI
furrow web
```

## Configuration

Furrow is configured via environment variables (or a `.env` file). See
[`.env.example`](.env.example) for a complete reference.

| Variable                   | Description                          | Default                  |
| -------------------------- | ------------------------------------ | ------------------------ |
| `FURROW_PROVIDER`          | LLM provider: `anthropic`\|`openai`\|`ollama` | `anthropic`      |
| `FURROW_MODEL`             | Default chat model                   | `claude-sonnet-4-20250514` |
| `FURROW_PLANNER_MODEL`     | Planner subagent model               | `claude-3-5-haiku-20241022` |
| `FURROW_WORKER_MODEL`      | Worker subagent model                | `claude-3-5-sonnet-20241022` |
| `FURROW_TESTER_MODEL`      | Tester subagent model                | `claude-3-5-sonnet-20241022` |
| `FURROW_ANTHROPIC_API_KEY` | Anthropic API key                    | —                        |
| `FURROW_OPENAI_API_KEY`    | OpenAI API key                       | —                        |
| `FURROW_OLLAMA_BASE_URL`   | Ollama base URL (for local LLM)      | `http://localhost:11434` |
| `FURROW_MAX_PARALLEL_TASKS`| Max parallel tasks per cycle         | `5`                      |
| `FURROW_MAX_CYCLES`        | Max development cycles (0 = unlimited)| `0`                   |
| `FURROW_WORKSPACE`         | Project workspace directory          | `.`                      |
| `FURROW_WEB_HOST`          | Web server bind host                 | `0.0.0.0`                |
| `FURROW_WEB_PORT`          | Web server bind port                 | `8000`                   |
| `FURROW_LOG_LEVEL`         | Log level (`DEBUG`\|`INFO`\|...)     | `INFO`                  |
| `FURROW_LLM_TIMEOUT`       | LLM request timeout (seconds)        | `120`                    |
| `FURROW_LLM_MAX_RETRIES`   | Max LLM retry attempts               | `3`                      |
| `FURROW_TEST_TIMEOUT`      | Test suite timeout (seconds)         | `120`                    |

## Commands

```bash
# Run interactively (prompts for a goal if none is provided)
furrow start "Deploy the API to staging"
furrow start

# Override the LLM model for a single run
furrow start "Refactor the auth module" --model claude-3-5-sonnet-20241022

# Start the web monitoring server
furrow web

# Run as a Python module
python -m furrow start "Build a CLI todo app"
```

### `/furrow` slash command

Inside Kilo, run `/furrow <goal>` to launch the orchestrator directly in an agent session.

## Architecture

Furrow is composed of a primary orchestrator and three specialized subagents:

```
furrow/
├── __main__.py          # python -m furrow entry point
├── __init__.py
├── config.py            # Settings, models (Plan, TaskModel, TestResult)
├── llm.py               # LLM client with retry logic
├── cli/
│   ├── __init__.py
│   └── main.py          # Click CLI (start, web)
├── core/
│   ├── __init__.py      # Public exports
│   ├── state.py         # Persistent session state (StateManager)
│   └── orchestrator.py  # Development loop: plan → execute → test → repeat
├── agents/
│   ├── planner.py       # Breaks goals into parallel tasks
│   ├── worker.py        # Executes a single task
│   └── tester.py        # Runs tests & reports results
└── web/
    ├── __init__.py
    └── server.py        # FastAPI web UI for monitoring
```

### The Development Loop

1. **Plan** — The planner subagent decomposes the current goal into 1–5 independent
   parallelizable tasks.
2. **Execute** — The worker subagent runs each task concurrently (up to
   `FURROW_MAX_PARALLEL_TASKS`).
3. **Test** — The tester subagent runs the project's test suite, linter, and type checker.
4. **Iterate** — If tests fail, the failing cases become the next cycle's goal. If all
   tasks succeed and tests pass, the loop halts.

### Persistent State

Session state is persisted to `<workspace>/.furrow/state.json` so that goals, task lists,
plan/test histories, and errors survive across runs. The `StateManager` in
`furrow/core/state.py` reads and writes this file.

## Development

```bash
# Install with dev dependencies
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
