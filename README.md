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

Set environment variables in a `.env` file (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `FURROW_PROVIDER` | LLM provider: `anthropic`, `openai`, or `ollama` | `anthropic` |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `FURROW_OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `FURROW_MODEL` | Default model | `claude-sonnet-4-20250514` |
| `FURROW_PLANNER_MODEL` | Model for planning | `claude-3-5-haiku-20241022` |
| `FURROW_WORKER_MODEL` | Model for code execution | `claude-3-5-sonnet-20241022` |
| `FURROW_TESTER_MODEL` | Model for test evaluation | `claude-3-5-sonnet-20241022` |
| `FURROW_MAX_PARALLEL_TASKS` | Max concurrent tasks | `5` |
| `FURROW_MAX_CYCLES` | Max cycles before halt (`0` = unlimited) | `0` |
| `FURROW_LOG_LEVEL` | Log level | `INFO` |

## Architecture

Furrow runs a continuous loop of **Plan → Execute → Test**:

1. **Plan**: A lightweight planner model breaks the goal into 1-5 independent tasks.
2. **Execute**: Worker agents run tasks in parallel, using file tools to read/write code.
3. **Test**: A tester agent runs the project's test suite and evaluates results.
4. **Repeat**: If tests fail, the goal is updated with failure details and the cycle continues.

## Web UI

```bash
furrow web
```

Open http://localhost:8000 to use the web interface with live WebSocket streaming.