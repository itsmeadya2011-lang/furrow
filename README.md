# Furrow

Furrow is an autonomous coding agent that plans, implements, and tests changes to a codebase through an iterative loop. It coordinates a planner, a worker (which edits files using real tool calls), and a tester to move toward a goal while persisting state and optionally streaming progress via a web UI.

## Features

- Plans tasks with dependency tracking and parallel execution
- Worker actually implements code using file and command tools
- Project-aware test detection (pytest, npm/pnpm/yarn, cargo, go)
- Event-driven progress with a console UI and FastAPI web interface (readable event log with timestamps, status header, and per-task panels)
- Atomic state persistence with resume support
- Offline test suite with no API keys required

## Security

The worker's command execution is hardened: commands are normalized (lowercased, whitespace collapsed), split into segments on `;`, `&&`, `||`, `|`, and newlines, and every segment is checked. Destructive programs by name (`mkfs*`, `fdisk`, `parted`, `shutdown`, `reboot`, `poweroff`, `halt`, `init`, `dd`) and destructive patterns (`rm -rf /`, `rm -fr /`, `rm -rf /*`, `rm -rf ~`, `chmod -R 777 /`, `> /dev/sda`, the fork bomb) are denied. Force-pushing (`git push --force`, `-f`, `--force-with-lease`) is blocked, and redirection to an absolute path outside the workspace is denied. Ordinary development commands continue to work.

## Install

Install in editable mode with dev dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### `furrow start "<goal>"`
Start (or resume) a run for the specified goal.

```bash
furrow start "Add JWT auth to the API"
```

Options:
- `--model` — override the LLM model
- `--max-cycles N` — maximum number of cycles (0 = unlimited)
- `--workspace PATH` — project directory to work in
- `--resume` — resume the persisted run instead of starting fresh

If `goal` is omitted, Furrow will prompt for it.

### `furrow status`
Show the persisted state of the last run for a workspace.

```bash
furrow status
furrow status --workspace /path/to/project
```

The status command shows the original goal, current goal, cycle count, last update time, token usage (if any), and per-task entries including which files each task changed, if any.

### `furrow resume`
Continue the persisted run.

```bash
furrow resume
furrow resume --workspace /path/to/project --max-cycles 5
```

### `furrow web`
Serve the web UI and stream orchestrator events.

```bash
furrow web
furrow web --host 127.0.0.1 --port 8000
```

The default web UI is available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

The web API refuses to start a second run while one is already in progress — both through `POST /api/start` and through the websocket.

## Configuration

Furrow reads environment variables with the `FURROW_` prefix (and a `.env` file if present). Available settings:

| Variable | Default | Description |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, or `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model for completions/chat |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model for the planner |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model for the worker |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model for the tester |
| `FURROW_ANTHROPIC_API_KEY` | (none) | Anthropic API key |
| `FURROW_OPENAI_API_KEY` | (none) | OpenAI API key |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama OpenAI-compatible base URL |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Maximum concurrent worker tasks per wave |
| `FURROW_MAX_CYCLES` | `0` | Maximum cycles to run (0 = unlimited) |
| `FURROW_MAX_WORKER_ITERATIONS` | `12` | Maximum tool-call iterations per worker task |
| `FURROW_MAX_RETRIES` | `3` | Retry attempts for transient LLM errors |
| `FURROW_MAX_TASK_RETRIES` | `1` | Times a failed task is retried before it is marked failed |
| `FURROW_LOG_LEVEL` | `INFO` | Logging level used by the CLI (honoured at startup) |
| `FURROW_REQUEST_TIMEOUT` | `120.0` | Request timeout in seconds |
| `FURROW_MAX_TOKENS` | `4096` | Max tokens for LLM completions |
| `FURROW_STATE_FILE` | `.furrow/state.json` | State file path (relative to workspace) |
| `FURROW_WORKSPACE` | current directory | Working directory for the agent |

A `.env` file is read automatically via pydantic-settings.

## How the loop works

1. **Plan** — The planner creates a `Plan` with tasks, each task specifying `id`, `description`, `files`, and `dependencies`.
2. **Execute** — Tasks run in dependency waves. The orchestrator respects `max_parallel_tasks` and marks tasks as pending/running/completed/failed/blocked. A failed task is retried with the same worker up to `max_task_retries` times; only when the retries are exhausted is it marked failed (and its dependents blocked). A worker that exhausts its iteration budget without solving the task is also marked failed — running out of iterations is never reported as success. Commands run by the worker and tester are executed with a timeout; when a command times out, its whole process group is killed so no orphan processes are left behind.
3. **Test** — The tester detects the appropriate test command for the project (pytest, npm/pnpm/yarn, cargo, go) and returns a `TestResult`. When no test runner is detectable, the result is `skipped` and the run is reported as `SKIPPED` rather than passed.
4. **Repeat** — If tests fail, the goal is amended with failure details to guide the next cycle. `max_cycles` controls termination (0 = unlimited). The worker uses a real tool-calling loop with `read_file`, `write_file`, `list_files`, and `run_command` to actually modify files in the sandboxed workspace.

## LLM token accounting

The LLM client records input/output tokens and call counts for every completion, and the orchestrator accumulates them per cycle and per run. Usage is persisted in the run state file and printed by the CLI in the run summary and in `furrow status`. Only token counts are tracked — cost estimation in currency is not implemented yet.

## State & resume

State is persisted atomically after every cycle to `<workspace>/.furrow/state.json`. Use `furrow resume` or `furrow start --resume` to continue a previous run. Corrupt state files are ignored and the run starts fresh. On resume, the persisted state file is loaded exactly once, so restored values cannot be overwritten by a second read.

## Development

```bash
cd /workspace/app
python -m pytest -q           # run the full offline test suite
python -m ruff check .        # linting
python -m mypy furrow         # type checking
```

All tests run offline and do not require API keys.