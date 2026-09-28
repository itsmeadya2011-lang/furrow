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

## Configuration

Furrow is configured via environment variables (prefixed with `FURROW_`) or a
`.env` file in the working directory.

| Variable | Description | Default |
|---|---|---|
| `FURROW_PROVIDER` | LLM provider: `anthropic`, `openai`, or `ollama` | `anthropic` |
| `FURROW_MODEL` | Default completion model | `claude-sonnet-4-20250514` |
| `FURROW_PLANNER_MODEL` | Model used by the planner | `claude-3-5-haiku-20241022` |
| `FURROW_WORKER_MODEL` | Model used by workers | `claude-3-5-sonnet-20241022` |
| `FURROW_TESTER_MODEL` | Model used by the tester | `claude-3-5-sonnet-20241022` |
| `FURROW_MAX_CYCLES` | Maximum development cycles (`0` = unlimited) | `0` |
| `FURROW_WORKSPACE` | Working directory | current directory |

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (or both) in your environment.

### Ollama

Run a local Ollama server and point Furrow at it:

```bash
export FURROW_PROVIDER=ollama
export FURROW_OLLAMA_BASE_URL=http://localhost:11434
export FURROW_OLLAMA_MODEL=llama3
```

### State persistence

During a run, Furrow writes its goal, cycle count, task statuses, and last test
result to `.furrow/state.json`. If the run is interrupted, the next invocation
resumes from the saved state.
