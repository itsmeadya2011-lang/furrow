# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Features

- **Infinite development loop**: Plans, executes, tests, and repeats until the goal is complete or stopped.
- **Parallel execution**: Breaks goals into 1-5 independent tasks that run concurrently.
- **Multiple LLM providers**: Anthropic, OpenAI, and Ollama support out of the box.
- **CLI and Web UI**: Run from the terminal or monitor progress in a browser.

## Installation

```bash
pip install furrow
```

## Usage

### CLI

Start the autonomous loop from the terminal:

```bash
furrow start "Add JWT auth to the API"
```

Optional flags:

```bash
furrow start --model gpt-4o --provider openai --max-cycles 10 --max-parallel 3 "Refactor the database layer"
```

- `--model`: Override the LLM model
- `--provider`: Choose `anthropic`, `openai`, or `ollama`
- `--max-cycles`: Limit the number of cycles (0 = unlimited)
- `--max-parallel`: Limit concurrent tasks per cycle (default: 5)

### Web UI

Launch a browser-based interface:

```bash
furrow web
```

Then open `http://localhost:8000`. The Web UI streams cycle output, lets you pick a provider, and shows pass/fail status in real time.

## Configuration

Furrow uses [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) with an `FURROW_` prefix. Set environment variables or create a `.env` file in your project root.

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `FURROW_PROVIDER` | LLM provider: `anthropic`, `openai`, or `ollama` | `anthropic` |
| `FURROW_MODEL` | Default model for the orchestrator | `claude-sonnet-4-20250514` |
| `FURROW_PLANNER_MODEL` | Model used by the planner agent | `claude-3-5-haiku-20241022` |
| `FURROW_WORKER_MODEL` | Model used by worker agents | `claude-3-5-sonnet-20241022` |
| `FURROW_TESTER_MODEL` | Model used by the tester agent | `claude-3-5-sonnet-20241022` |
| `FURROW_ANTHROPIC_API_KEY` | Anthropic API key | |
| `FURROW_OPENAI_API_KEY` | OpenAI API key | |
| `FURROW_OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `FURROW_MAX_PARALLEL_TASKS` | Max tasks per cycle | `5` |
| `FURROW_MAX_CYCLES` | Max total cycles (0 = unlimited) | `0` |
| `FURROW_MAX_RETRIES` | Retry attempts for LLM calls | `3` |
| `FURROW_WORKSPACE` | Project workspace directory | Current working directory |
| `FURROW_LOG_LEVEL` | Logging verbosity | `INFO` |
| `FURROW_MAX_TOKENS` | Max tokens per LLM completion | `4096` |
| `FURROW_TEST_COMMANDS` | Comma-separated test runner commands | Auto-detected |

### Provider Setup

**Anthropic**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
furrow start "Build a REST API"
```

**OpenAI**

```bash
export OPENAI_API_KEY="sk-..."
furrow start --provider openai "Build a CLI tool"
```

**Ollama**

```bash
export FURROW_PROVIDER=ollama
export FURROW_OLLAMA_BASE_URL=http://localhost:11434
furrow start "Refactor utils"
```

## Architecture

Furrow runs an autonomous planning-execution-testing loop using four specialized agents.

### Orchestrator

The `Orchestrator` runs the main loop:

1. **Plan** - Calls the planner to break the goal into 1-5 parallel tasks.
2. **Execute** - Spawns worker agents concurrently, respecting `max_parallel_tasks`.
3. **Test** - Runs the project's test suite via the tester agent.
4. **Report** - Summarizes results and repeats until done.

The orchestrator maintains an `OrchestratorState` object that tracks:
- Cycle history (plan + test results per cycle)
- Failed tasks for retry context
- Goal modifications when tests fail

### Planner (`PlannerAgent`)

Breaks a high-level goal into independent, parallelizable tasks. Returns a `Plan` JSON containing `tasks[]`, each with `id`, `description`, `files`, and `dependencies`.

### Worker (`WorkerAgent`)

Executes a single assigned task. Reads relevant files for context, makes minimal targeted changes, and returns a summary. Workers are scoped to one task to avoid scope creep.

### Tester (`TesterAgent`)

Runs the project's test commands (pytest, npm test, cargo test, etc.), captures output, and uses the LLM to classify pass/fail. Returns a structured `TestResult`.

### LLM Client (`LLMClient`)

Unified async client for all three providers. Includes automatic retries with exponential backoff for transient errors (rate limits, connection drops, 5xx responses). Tracks token usage per call.

```
furrow/
  __init__.py
  config.py          # Settings, Provider enum, data models
  llm.py             # LLMClient with Anthropic/OpenAI/Ollama support
  core/
    orchestrator.py  # Orchestrator + OrchestratorState
  agents/
    planner.py       # PlannerAgent
    worker.py        # WorkerAgent
    tester.py        # TesterAgent
    prompts.py       # Agent system prompts
  cli/
    main.py          # Click CLI: start, web
  web/
    server.py        # FastAPI + WebSocket UI
```

## Development Setup

```bash
git clone https://github.com/your-org/furrow.git
cd furrow
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Testing

```bash
pytest
```

## Linting and Type Checking

```bash
ruff check .
ruff format .
mypy furrow
```
