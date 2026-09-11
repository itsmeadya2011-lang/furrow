# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow web
```

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.

## Configuration

Environment variables (prefix `FURROW_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `FURROW_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, or `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model for planning |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model for workers |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model for testing |
| `FURROW_MAX_TOKENS` | `4096` | Max tokens per LLM request |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Max parallel worker tasks |
| `FURROW_MAX_CYCLES` | `0` | Max planning cycles (0 = infinite) |
| `FURROW_WORKSPACE` | `cwd` | Working directory |
| `FURROW_LOG_LEVEL` | `INFO` | Logging level |

## Architecture

Furrow runs in cycles:
1. **Plan**: Breaks the goal into parallel tasks
2. **Execute**: Runs workers in parallel to implement tasks
3. **Test**: Runs tests and evaluates results
4. **Repeat**: Fixes failures in the next cycle if needed

The web interface provides a simple WebSocket-based UI for monitoring progress.
