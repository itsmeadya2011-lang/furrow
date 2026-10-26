# Furrow

Furrow is an autonomous coding agent that plans, implements, and tests changes to a codebase through an iterative loop. It coordinates a planner, a worker (which edits files using real tool calls), and a tester to move toward a goal while persisting state and optionally streaming progress via a web UI.

## Features

- Plans tasks with dependency tracking and parallel execution
- Worker actually implements code using file and command tools
- Project-aware test detection (pytest, npm/pnpm/yarn, cargo, go)
- Event-driven progress with a console UI and FastAPI web interface (readable event log with timestamps, status header, and per-task panels)
- Finished work is not redone: a task that already completed is skipped when the next plan describes the same work
- Bounded event queues: a subscriber that stops draining loses its oldest events, never the newest ones
- Atomic state persistence with resume support
- Fail-safe cycle loop: a planner, execution, tester or state-save failure is reported as an ERROR event instead of crashing the run
- Hardened provider calls: transient HTTP errors are retried when the status is exposed on the response, and a malformed provider response raises a clear error
- Offline test suite with no API keys required

## Security

The worker's command execution is hardened: commands run through `/bin/sh -c` so ordinary shell constructs (`cd dir && pytest`, pipes, variable assignments, redirections) work. Commands are normalized (lowercased, whitespace collapsed, home variables canonicalised), split into segments on `;`, `&&`, `||`, `|`, and newlines, and every segment and every command substitution (`$(...)`, backticks) is checked. Destructive programs by name (`mkfs*`, `dd`, `fdisk`, `parted`, `shutdown`, `reboot`, `poweroff`, `halt`, `init`) and destructive patterns (`rm -rf /` in all spellings including `rm -rf "$HOME"`, `rm -fr /`, `rm -rf /*`, `chmod -R 777 /`, the fork bomb, `> /dev/sda`) are denied. Force-pushing (`git push --force`, `-f`, `--force-with-lease`) is blocked, and any redirection whose target resolves outside the workspace (absolute paths, `..` traversal, `~` and `$HOME` targets) is denied; redirections inside the workspace are allowed. Commands run in their own process group so a timeout kills the whole tree. The worker's tools treat their arguments defensively as well: a tool argument value that is not an object is reported back as an `error: invalid tool arguments` tool result instead of failing the whole task. Ordinary development commands continue to work.

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
- `--max-cycles N` — maximum number of cycles (0 = unlimited; negative values are rejected)
- `--workspace PATH` — project directory to work in
- `--resume` — resume the persisted run instead of starting fresh

If `goal` is omitted, Furrow will prompt for it.

The command exits with status `1` when the run ends `FAILED` (tests ran and did not pass) and with status `0` for `PASSED` (all tests passed), `SKIPPED` (no test runner detected) and `NOT RUN` (no tests were executed).

### `furrow status`
Show the persisted state of the last run for a workspace.

```bash
furrow status
furrow status --workspace /path/to/project
```

The status command shows the original goal, current goal, cycle count, last update time, token usage (if any), and per-task entries including which files each task changed, if any. The run summary includes completed/failed/blocked task counts, LLM usage, files changed, and a result line: `Result: PASSED - all tests passed.`, `Result: FAILED - tests did not pass.`, `Result: SKIPPED - no test runner detected.`, or `Result: NOT RUN - no tests were executed.` when a run completed without executing any tests.

### `furrow resume`
Continue the persisted run.

```bash
furrow resume
furrow resume --workspace /path/to/project --max-cycles 5
```

The exit codes are the same as for `furrow start`: `1` when the resumed run ends `FAILED`, `0` for `PASSED`, `SKIPPED` and `NOT RUN`.

### `furrow web`
Serve the web UI and stream orchestrator events.

```bash
furrow web
furrow web --host 127.0.0.1 --port 8000
```

The default web UI is available at [http://127.0.0.1:8000](http://127.0.0.1:8000). If the server cannot start, the command reports a friendly error and exits with status 1. The status header (goal, cycle, state, stopped reason, usage) is bound correctly and is refreshed by status fetches triggered by websocket events (task started, task finished, done) plus a single fetch on page load; the client does not poll on a timer. The run summary is appended to the event log only when it changes.

A task status comes from the LLM-planned task, so it is untrusted input: the UI maps it through a fixed set of known statuses (`pending`, `running`, `completed`, `failed`, `blocked`, `skipped`) with an `unknown` fallback before it is used in a CSS class name, and the status text itself is still written with `textContent`.

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

The numeric settings are validated: `max_parallel_tasks` >= 1, `max_cycles` >= 0 (0 still means unlimited), `max_retries` >= 0, `max_task_retries` >= 0, `max_worker_iterations` >= 1, `max_tokens` >= 1, and `request_timeout`, `command_timeout` and `test_timeout` must be greater than 0. An out-of-range value is rejected when the settings are built for a run, and `--max-cycles` rejects a negative value at the command line.

## How the loop works

1. **Plan** — The planner creates a `Plan` with tasks, each task specifying `id`, `description`, `files`, and `dependencies`.
2. **Execute** — Tasks run in dependency waves. The orchestrator respects `max_parallel_tasks` and marks tasks as pending/running/completed/failed/blocked. A failed task is retried with the same worker up to `max_task_retries` times; only when the retries are exhausted is it marked failed (and its dependents blocked). A plan merge preserves a task's terminal status for reporting — a task that failed with an exhausted retry budget stays failed, a blocked task stays blocked — and preserves its attempt counter; the retry budget is enforced within a cycle, not across cycles. A task that already completed is not run again when the new plan describes the same work (the same description and the same files): it is skipped, no attempt is consumed, its status stays `completed`, and a log event records the skip. When the planner re-plans the same task id with a different description or different files, the task counts as new work: its status is reset to `pending` and it is executed again. A merge never wipes a result or an error that was already recorded for a task. A worker that exhausts its iteration budget without solving the task is also marked failed — running out of iterations is never reported as success. Commands run by the worker and tester are executed with timeouts from `command_timeout` and `test_timeout` respectively; when a command times out, its whole process group is killed so no orphan processes are left behind. The worker executes commands via `/bin/sh -c` with a hardened guard (segments, command substitutions, denied programs/patterns, force-pushes, and workspace-scoped redirections). The tester treats the exit status as ground truth (non-zero exit is failure even if the LLM reports passed; a runner that cannot start is an infrastructure error; `test_timeout` controls the test suite; "0 failed"/"no failures" are not misread as failures). The no-progress guard counter and the current plan signature are persisted in run state and restored on resume, so a resumed run does not restart the guard from zero; the guard is evaluated before the state is written, so the persisted counter is the current one instead of lagging the live one by a cycle (and the persisted signature is the current cycle's). The state file is written exactly once per completed cycle, with a single `STATE_SAVED` event. Tasks blocked by failed dependencies remain blocked and are reported separately. `RunResult` includes `tasks_blocked` and `tests_run`. `Settings` adds `command_timeout` (default 300s) and `test_timeout` (default 600s).
3. **Test** — The tester detects the appropriate test command for the project (pytest, npm/pnpm/yarn, cargo, go) and returns a `TestResult`. When no test runner is detectable, the result is `skipped` and the run is reported as `SKIPPED` rather than passed.
4. **Repeat** — If tests fail, the goal is amended with failure details to guide the next cycle. `max_cycles` controls termination (0 = unlimited). The worker uses a real tool-calling loop with `read_file`, `write_file`, `list_files`, and `run_command` to actually modify files in the sandboxed workspace.

The loop is fail-safe: a planner error, an execution error, a tester error or a state-save error no longer crashes the run. A planner, execution or tester error emits an ERROR event, sets a stopped reason (`planner failed: ...`, `execute failed: ...`, `tester failed: ...`) and ends the loop cleanly with a normal run result; a state-save error emits an ERROR event with the path that could not be written and the run keeps going. Cancellation (`asyncio.CancelledError`) still propagates. An exception raised while handling a worker's result is converted into a task failure that consumes one attempt — it used to be swallowed by `asyncio.gather(..., return_exceptions=True)` — so a task is never left in an inconsistent state.

## LLM token accounting

The LLM client records input/output tokens, call counts and an estimated USD cost for every completion, and the orchestrator accumulates them per cycle and per run. Usage is persisted in the run state file and printed by the CLI in the run summary and in `furrow status`. The estimated cost is looked up per model from a built-in snapshot of published per-1,000,000-token USD prices (longest-prefix match, provider prefixes stripped) and can be overridden or extended with the `FURROW_MODEL_PRICING` environment variable, which takes a JSON object mapping model ids to prices::

    FURROW_MODEL_PRICING='{"my-model":{"input_per_million":1.0,"output_per_million":2.0}}'

The prices are a snapshot taken at a point in time and go stale as providers change them; they are an approximation, and an unknown model reports a cost of `0.00`.

The provider clients (Anthropic, OpenAI, Ollama) are created lazily and released when a run ends: `LLMClient.aclose()` closes the ones that were created and leaves the client usable afterwards. The CLI releases the client when a run ends — including when the run raises — and the web server releases it when a run finishes, errors or is cancelled, so a long-running server no longer leaks a client per run.

## State & resume

State is persisted atomically at the end of every completed cycle to `<workspace>/.furrow/state.json`; a cycle that ends early (a planner error, or a plan with no tasks) is not persisted, so the saved cycle count can lag the reported one by one. Each completed cycle writes the state exactly once, after the no-progress guard has been evaluated, so the saved `no_progress` counter and plan signature are the current ones. A `STATE_SAVED` event is emitted only when the state file was actually written; a failed write is reported as an ERROR event with the path that could not be written instead. The state file also carries the number of test-suite executions (`tests_run`); it is restored on resume, so `RunResult.tests_run` counts the whole run and not only the cycles since the resume. Use `furrow resume` or `furrow start --resume` to continue a previous run. Corrupt state files are ignored and the run starts fresh. On resume, the persisted state file is loaded exactly once, so restored values cannot be overwritten by a second read.

## Development

```bash
cd /workspace/app
python -m pytest -q           # run the full offline test suite
python -m ruff check .        # linting
python -m mypy furrow         # type checking
```

All tests run offline and do not require API keys.