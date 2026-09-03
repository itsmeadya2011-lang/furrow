# Furrow

**Autonomous coding agent with an infinite parallel development loop.**

Furrow is an agentic coding system that repeatedly plans, writes, and tests code in parallel until a goal is achieved or the cycle limit is reached. It supports multiple LLM providers, a multi-agent architecture, and can run from the CLI or a web UI.

---

## Features

- **Parallel development loop** – multiple tasks execute concurrently each cycle, with the orchestrator retrying failed plans until success or cycle exhaustion.
- **Multi-agent architecture** – a Planner, Worker, and Tester agent coordinate with the Orchestrator across cycles.
- **Multi-provider support** – works with Anthropic, OpenAI, and Ollama.
- **CLI and web UI** – run agents from the command line or drive them through a browser-based dashboard over WebSocket.

---

## Installation

```bash
pip install furrow
```

For development (see [Development](#development)):

```bash
pip install -e ".[dev]"
```

---

## Quick Start

### CLI

```bash
furrow start "Add JWT auth to the API"
```

Control the maximum number of cycles (default `0` = infinite):

```bash
furrow start --cycles 5 "Refactor the database layer"
```

Override the LLM model for a single run:

```bash
furrow start --model gpt-4o "Fix the login flow"
```

### Web UI

```bash
furrow web
```

This starts a [FastAPI](https://fastapi.tiangolo.com/) server with a [Uvicorn](https://www.uvicorn.org/) dev server on `http://localhost:8000`. Open the page in a browser, enter a goal, and watch real-time output streamed over WebSocket.

### Environment Variables

Set your API key(s) **before** running Furrow:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"     # only if using the OpenAI provider
```

> When using the Ollama provider, no API key is required; configure the endpoint with `FURROW_OLLAMA_BASE_URL` (default `http://localhost:11434`).

---

## Configuration

All configuration is handled by the `Settings` class in `furrow/config.py` (powered by [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)). Values can be set via environment variables prefixed with `FURROW_`, or in a `.env` file in your workspace.

### Provider Selection

| Variable | Description | Default |
|---|---|---|
| `FURROW_PROVIDER` | LLM provider: `anthropic`, `openai`, or `ollama` | `anthropic` |

### Model Overrides

Each agent role can use a different model. The global `model` acts as the default; the role-specific keys override it for that agent only.

| Variable | Description | Default |
|---|---|---|
| `FURROW_MODEL` | Default model used across agents | `claude-sonnet-4-20250514` |
| `FURROW_PLANNER_MODEL` | Model used by the Planner agent | `claude-3-5-haiku-20241022` |
| `FURROW_WORKER_MODEL` | Model used by the Worker agent | `claude-3-5-sonnet-20241022` |
| `FURROW_TESTER_MODEL` | Model used by the Tester agent | `claude-3-5-sonnet-20241022` |

### API Keys

| Variable | Description |
|---|---|
| `FURROW_ANTHROPIC_API_KEY` | Anthropic API key (or set `ANTHROPIC_API_KEY`) |
| `FURROW_OPENAI_API_KEY` | OpenAI API key (or set `OPENAI_API_KEY`) |
| `FURROW_OLLAMA_BASE_URL` | Ollama endpoint URL | `http://localhost:11434` |

### Execution Controls

| Variable | Description | Default |
|---|---|---|
| `FURROW_MAX_PARALLEL_TASKS` | Maximum number of Worker tasks run concurrently per cycle | `5` |
| `FURROW_MAX_CYCLES` | Maximum number of development cycles. `0` = infinite. | `0` |

### Workspace & Logging

| Variable | Description | Default |
|---|---|---|
| `FURROW_WORKSPACE` | Path used as the working directory for file operations | Current working directory |
| `FURROW_LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |

---

## Architecture

Furrow operates in repeated **cycles**, each driven by the **Orchestrator** and a team of four coordinated agents:

```
Goal ──► ┌────────────────────┐
         │  Orchestrator      │  (cycle loop + max_cycles control)
         └────────┬───────────┘
                  │
   ┌──────────────┼──────────────────┐
   │              │                  │
   ▼              ▼                  ▼
Planner      →  Worker(s)        Tester
(plan tasks)     (implement)      (run tests)
```

1. **Orchestrator** – drives the development loop. Each cycle calls the Planner, runs all Worker tasks in parallel, then hands results to the Tester. If tests fail, the failure summary becomes the goal for the next cycle. The loop halts when all tasks pass or `MAX_CYCLES` is reached.

2. **Planner** – analyzes the goal and the current workspace, then produces a `Plan` containing an ordered list of `Task` objects (each with a description, target files, and optional dependencies).

3. **Worker** – executes a single task by generating and applying code changes directly in the workspace using role-specific LLM prompts.

4. **Tester** – runs the project's test suite (auto-detecting `pytest`, `npm`/`pnpm`/`yarn`, `cargo`, or `go`), then asks an LLM to evaluate the output and return a `TestResult` indicating pass/fail and any failures.

Each agent uses the `LLMClient`, which abstracts over the selected provider and adds automatic retries with exponential backoff via [tenacity](https://www.tenacity.readthedocs.io/).

---

## Development

### Setup

```bash
git clone https://github.com/your-org/furrow.git
cd furrow
pip install -e ".[dev]"
```

### Test

```bash
pytest
```

### Lint

```bash
ruff check .
```

### Type Check

```bash
mypy furrow
```

---

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
