# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Features

- **Infinite development loop**: Continuously plans, implements, tests, and iterates until the task is complete.
- **Parallel workers**: Runs multiple workers concurrently to tackle independent subtasks.
- **Planner/Tester agents**: Dedicated agents for task planning and automated testing.
- **Web UI**: Monitor progress and manage tasks through a built-in web interface.

## Install

```bash
pip install furrow
```

## Requirements

- Python >= 3.10

## CLI Commands

### `furrow start <task>`

Start the autonomous development loop with a natural language task description.

```bash
furrow start "Add JWT auth to the API"
```

### `furrow web`

Launch the web UI to monitor active tasks and review results.

```bash
furrow web
```

## Configuration

Furrow is configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `FURROW_PROVIDER` | LLM provider: `anthropic`, `openai`, or `ollama` | `anthropic` |
| `FURROW_MODEL` | Default model identifier | provider default |
| `FURROW_PLANNER_MODEL` | Model for the planner agent | same as `FURROW_MODEL` |
| `FURROW_WORKER_MODEL` | Model for worker agents | same as `FURROW_MODEL` |
| `FURROW_TESTER_MODEL` | Model for tester agents | same as `FURROW_MODEL` |
| `FURROW_ANTHROPIC_API_KEY` | Anthropic API key | |
| `FURROW_OPENAI_API_KEY` | OpenAI API key | |
| `FURROW_OLLAMA_BASE_URL` | Base URL for Ollama | `http://localhost:11434` |
| `FURROW_MAX_PARALLEL_TASKS` | Maximum number of parallel workers | `4` |
| `FURROW_MAX_CYCLES` | Maximum development loop cycles | `10` |
| `FURROW_LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |

Example:

```bash
export FURROW_PROVIDER=openai
export FURROW_MODEL=gpt-4o
export FURROW_OPENAI_API_KEY=sk-...
export FURROW_MAX_PARALLEL_TASKS=2
```
