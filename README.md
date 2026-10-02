# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
# Run with a goal
furrow start "Add JWT auth to the API"

# Start the web UI
furrow web
```

### CLI Options

The `start` command supports several options:

```bash
furrow start "Add JWT auth to the API" \
    --provider anthropic \
    --model claude-sonnet-4-20250514 \
    --max-cycles 5 \
    --workspace /path/to/project
```

- `--model`: Override the LLM model used for all agents (planner, worker, tester).
- `--provider`: Choose `anthropic`, `openai`, or `ollama`.
- `--max-cycles`: Limit the number of planning/execution cycles (0 = unlimited).
- `--workspace`: Set the workspace directory to operate on.

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment. For Ollama,
run `ollama serve` locally (or set `FURROW_OLLAMA_BASE_URL`).

## Architecture

Furrow runs an infinite loop:

1. **Plan** — the planner breaks the goal into 1–5 parallelizable tasks.
2. **Execute** — worker agents use LLM tool calling (read/write/list/edit files)
   to implement each task in parallel, respecting dependencies.
3. **Test** — the tester runs the project's test suite and reports results.
4. **Repeat** — if tests fail, the goal is updated with fix instructions and
   the loop continues.

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| Orchestrator | `furrow/core/orchestrator.py` | Runs the infinite loop with cycle/max_cycles limits |
| Planner | `furrow/agents/planner.py` | Breaks goals into parallel tasks via LLM |
| Worker | `furrow/agents/worker.py` | Executes tasks using LLM tool calling |
| Tester | `furrow/agents/tester.py` | Runs test suites and reports results |
| LLM Client | `furrow/llm.py` | Supports Anthropic, OpenAI, and Ollama with tool calling |

### Features

- **Multi-provider LLM support**: Anthropic, OpenAI, and Ollama
- **Tool calling**: Workers can read, write, list, and edit files
- **Dependency-aware execution**: Tasks run in topological batches
- **Parallel execution with limits**: Configurable `max_parallel_tasks`
- **Cycle limits**: `max_cycles` prevents runaway loops
- **Web UI**: Real-time WebSocket streaming of orchestrator output
- **Retry logic**: LLM API calls automatically retry with exponential backoff
- **Structured logging**: Uses `structlog` throughout
