# Furrow

Autonomous coding agent: give it a goal and it plans, implements in parallel, tests, and repeats until the work passes.

## Install

```bash
pip install furrow            # or: pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...  # or OPENAI_API_KEY; not needed for --provider ollama
```

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow start "Fix flaky tests" --provider ollama --model llama3 --max-cycles 0
furrow start --max-parallel 8 --test-command "pytest -q" --workspace ./app
furrow web --host 0.0.0.0 --port 8000
furrow version
```

`start` options (CLI flags override environment/config defaults):

| Option | Default | Notes |
| --- | --- | --- |
| `--model` | `claude-sonnet-4-20250514` | Model for every agent |
| `--provider` | `anthropic` | `anthropic`, `openai`, or `ollama` |
| `--max-cycles` | `10` | `0` means unlimited |
| `--max-parallel` | `5` | Max concurrent worker tasks |
| `--workspace` | cwd | Root for reads and edits |
| `--test-command` | auto-detected | Command run by the tester each cycle |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Omit the goal to be prompted for it interactively. `Ctrl-C` stops a run cleanly.

## Configuration

Settings come from `.env` or the environment via `FURROW_`-prefixed variables (pydantic-settings).

| Variable | Default | Notes |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | `anthropic`, `openai`, `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Overridden by `--model` |
| `FURROW_PLANNER_MODEL` / `FURROW_WORKER_MODEL` / `FURROW_TESTER_MODEL` | haiku / sonnet / sonnet | Per-role model overrides |
| `FURROW_ANTHROPIC_API_KEY` / `FURROW_OPENAI_API_KEY` | – | `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` also read |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `FURROW_MAX_CYCLES` | `10` | `0` means unlimited |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Worker concurrency limit |
| `FURROW_TEST_COMMAND` | – | Falls back to pytest/npm/cargo/go detection |
| `FURROW_TEST_TIMEOUT` | `120` | Seconds per test command |
| `FURROW_WORKSPACE` | cwd | Base directory for all file operations |
| `FURROW_LOG_LEVEL` | `INFO` | |

## Web UI

```bash
furrow web            # http://localhost:8000
```

Enter a goal in the browser; the orchestrator runs server-side and streams progress
events (`plan`, `task_start`, `edit`, `test`, `done`, ...) over a websocket. A
`GET /health` endpoint backs liveness checks.

## Architecture

- **Orchestrator** (`furrow/core/orchestrator.py`) — runs the plan → implement → test loop,
  bounds concurrency with a semaphore, and guards every edit against escaping the workspace.
- **Planner** — turns the goal plus recent history into a JSON list of independent tasks.
- **Workers** — run in parallel, each returning a summary and file edits that are written to disk.
- **Tester** — runs the test command, then asks the model to judge pass/fail from the output.
- **LLM client** (`furrow/llm.py`) — Anthropic/OpenAI/Ollama behind one interface, with retries
  and exponential backoff on transient API errors and tolerant JSON parsing (fences/prose).

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check furrow tests && ruff format furrow tests
mypy furrow
```
