# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
# Local development install
pip install -e .

# Or from PyPI (when published)
pip install furrow
```

## Configuration

Environment variables (prefix with `FURROW_` or use `.env` file):

| Variable | Description | Default |
|----------|-------------|---------|
| `FURROW_PROVIDER` | LLM provider: `anthropic`, `openai`, `ollama` | `anthropic` |
| `FURROW_MODEL` | Default model | `claude-sonnet-4-20250514` |
| `FURROW_PLANNER_MODEL` | Planner model | `claude-3-5-haiku-20241022` |
| `FURROW_WORKER_MODEL` | Worker model | `claude-3-5-sonnet-20241022` |
| `FURROW_TESTER_MODEL` | Tester model | `claude-3-5-sonnet-20241022` |
| `FURROW_ANTHROPIC_API_KEY` | Anthropic API key | — |
| `FURROW_OPENAI_API_KEY` | OpenAI API key | — |
| `FURROW_OLLAMA_BASE_URL` | Ollama base URL | `http://localhost:11434` |
| `FURROW_MAX_PARALLEL_TASKS` | Max concurrent tasks | `5` |
| `FURROW_MAX_CYCLES` | Max cycles (0 = unlimited) | `0` |
| `FURROW_WORKSPACE` | Working directory | current directory |
| `FURROW_LOG_LEVEL` | Log level | `INFO` |

## CLI Usage

```bash
# Basic usage (prompts for goal if omitted)
furrow start "Add JWT auth to the API"

# Override provider and models
furrow start "Refactor user service" \
  --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --planner-model claude-3-5-haiku-20241022 \
  --worker-model claude-3-5-sonnet-20241022 \
  --tester-model claude-3-5-sonnet-20241022

# Workspace and limits
furrow start "Fix flaky tests" \
  --workspace ./my-project \
  --max-cycles 10 \
  --max-parallel-tasks 3

# Web UI
furrow web
```

### `furrow start` Options

| Option | Description |
|--------|-------------|
| `GOAL` | Goal description (optional, will prompt) |
| `--provider` | Provider: `anthropic`, `openai`, `ollama` |
| `--model` | Default model override |
| `--planner-model` | Planner model override |
| `--worker-model` | Worker model override |
| `--tester-model` | Tester model override |
| `--workspace` | Working directory path |
| `--max-cycles` | Maximum cycles (0 = unlimited) |
| `--max-parallel-tasks` | Maximum parallel tasks |

## Web UI

```bash
furrow web
# Opens http://localhost:8000
```

The web interface provides a real-time view of Furrow's execution:
- Enter a goal and optional model/provider
- Live streaming of cycles, plans, tasks, and test results
- Color-coded events: cycle (teal), plan (blue), task (orange), test (green/red), complete (yellow), error (red)

### WebSocket API

Connect to `ws://localhost:8000/ws` and send:

```json
{
  "goal": "Add JWT auth to the API",
  "provider": "anthropic",
  "model": "claude-sonnet-4-20250514",
  "planner_model": "claude-3-5-haiku-20241022",
  "worker_model": "claude-3-5-sonnet-20241022",
  "tester_model": "claude-3-5-sonnet-20241022",
  "workspace": "/path/to/project",
  "max_cycles": 10,
  "max_parallel_tasks": 5
}
```

Receive JSON events:
```json
{"type": "cycle", "message": "Cycle 1 started", "data": {"cycle": 1}}
{"type": "plan", "message": "Plan updated", "data": {"plan": {...}, "tasks": [...]}}
{"type": "task", "message": "Task abc123 started", "data": {"task": {...}}}
{"type": "test", "message": "Tests passed: All green", "data": {"passed": true, "summary": "..."}}
{"type": "complete", "message": "Goal complete. Halting.", "data": {"completed": true}}
{"type": "error", "message": "Planning failed: ..."}
```

## Requirements

- Python 3.10+
- API key for chosen provider (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` or Ollama running locally)

## License

MIT
