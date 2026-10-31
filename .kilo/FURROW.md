# Furrow Architecture

This document describes both the **Kilo integration** (orchestrator/subagents/commands/config used in the Kilo environment) and the **Python `furrow/` package** (the standalone CLI/orchestrator/worker/tester/web stack). They are distinct layers that share the same name.

## Components

### Kilo integration (`.kilo/`)

The Kilo-side setup defines primary/subagents and a slash command for running Furrow inside Kilo:

| File | Type | Role |
| --- | --- | --- |
| `.kilo/agent/furrow.md` | Primary agent | Orchestrator for the infinite development loop. Coordinates planning, spawning workers, testing, and repeating. |
| `.kilo/agent/furrow-planner.md` | Subagent | Breaks a high-level goal into parallelizable tasks with IDs, descriptions, files, and dependencies. |
| `.kilo/agent/furrow-worker.md` | Subagent | Executes a single assigned task with targeted changes. |
| `.kilo/agent/furrow-tester.md` | Subagent | Runs tests/lint/type checks, fixes failures when possible, and returns pass/fail results. |
| `.kilo/command/furrow.md` | Slash command | Entry point (`/furrow <goal>`). Invokes the `furrow` orchestrator with `$ARGUMENTS`. |
| `kilo.json` | Project config | Sets the default agent to `furrow` and opens permissions for `bash`/`edit`/`read`. |

### Python package (`furrow/`)

| Module | Purpose |
| --- | --- |
| `furrow/config.py` | Pydantic models and settings. Enums: `Provider` (`anthropic`/`openai`/`ollama`), `TaskStatus`. Models: `TaskModel`, `Plan`, `TestResult` (`skipped`), `WorkerResult`, `RunResult` (`skipped`, `usage`, `tasks_blocked`, `tests_run`, `files_changed`), `Usage` (token accounting). `TaskModel.status` is normalised to the known status set (`pending`, `running`, `completed`, `failed`, `blocked`, `skipped`); an unknown status from untrusted planner output becomes `pending` instead of failing the whole plan. `Settings` uses `FURROW_` prefix, reads `.env`, honours `FURROW_LOG_LEVEL`; adds `command_timeout` (default 300s) and `test_timeout` (default 600s). The numeric settings are bounded (`max_parallel_tasks` >= 1, `max_cycles` >= 0 with 0 meaning unlimited, `max_retries` >= 0, `max_task_retries` >= 0, `max_worker_iterations` >= 1, `max_tokens` >= 1, and `request_timeout`/`command_timeout`/`test_timeout` > 0), so an out-of-range value is rejected when the settings are built. `Settings.state_path(workspace=None)` resolves `<workspace>/<state_file>`. |
| `furrow/llm.py` | Provider-neutral `LLMClient` with `complete()` and `chat(messages, tools, system, model)`. Converters for Anthropic/OpenAI message/tool formats. Supports Anthropic, OpenAI, and Ollama (OpenAI-compatible `/v1`). Tenacity retry on transient errors (408/409/425/429/5xx), non-retryable on 4xx. A transient provider error is recognised from `exc.response.status_code` (httpx-style errors, including the OpenAI/Anthropic `APIStatusError`) as well as from `exc.status_code`, so a 429/5xx response is actually retried. A malformed provider response raises a clear `ValueError` (`provider response contained no choices` / `provider response choice has no message`) instead of an obscure `IndexError`/`AttributeError`, and an Anthropic response without `content` is treated as empty text. Usage tracking: every `LLMResponse` carries a `usage` and `LLMClient.total_usage` accumulates input/output tokens and call counts across calls. `LLMClient.aclose()` closes the provider HTTP clients that were created and leaves the client usable. Helpers: `read_file`/`write_file`/`list_files`. |
| `furrow/agents/parsing.py` | Robust parsing: `strip_code_fences`, `extract_json`, `parse_json`, `parse_plan`, `parse_test_result`. Handles fences, prose around JSON, and nested structures. |
| `furrow/agents/planner.py` | `PlannerAgent.plan(goal, context="") -> Plan`. Retries once with stricter instructions on parse failure. |
| `furrow/agents/worker.py` | Core of the product. `WorkerAgent(task, client, settings, workspace, goal).run() -> WorkerResult`. Runs a real tool-calling loop with four tools: `read_file`, `write_file`, `list_files`, `run_command`. Paths are sandboxed to the workspace. Commands run via `/bin/sh -c` (supports `cd && ...`, pipes, assignments, redirections) in their own process group; timeout comes from `Settings.command_timeout` (default 300s) and kills the whole tree. The hardened guard normalizes the command, splits into segments, checks every segment and every command substitution (`$(...)`, backticks), and denies destructive programs by name (`mkfs*`, `dd`, `fdisk`, `parted`, `shutdown`, `reboot`, `poweroff`, `halt`, `init`), destructive patterns (`rm -rf /` in all spellings including `rm -rf "$HOME"`, `rm -fr /`, `rm -rf /*`, `chmod -R 777 /`, fork bomb, `> /dev/sda`), force pushes (`git push --force`, `-f`, `--force-with-lease`), and any redirection whose target resolves outside the workspace (absolute, `..`, `~`/`$HOME`); workspace redirections allowed. Stdout and stderr are combined into a single result string, with the stderr part behind a `[stderr]` marker; when the output is too long it keeps the tail and prefixes a `[... output truncated ...]` marker. Tool arguments are untrusted input and are coerced before use: an argument value that is not an object returns an `error: invalid tool arguments` tool result instead of raising and failing the whole task. `read_file` marks truncated content with a `[... output truncated ...]` marker too, and the directory listing runs off the event loop (`asyncio.to_thread`), so one big directory walk no longer stalls every parallel task. Iteration budget is `max_worker_iterations`. Exhausting the budget is a failure (never reported as success). |
| `furrow/agents/tester.py` | `TesterAgent.run(goal, tasks) -> TestResult`. `TesterAgent.detect_command() -> list[str] | None` detects per project type (pytest/npm/pnpm/yarn/cargo/go) instead of guessing binaries. Exit status is ground truth: non-zero exit is failure even if the LLM replies "passed"; a runner that cannot be started is an infrastructure error (distinct from a failing suite); test timeout comes from `Settings.test_timeout` (default 600s), separate from the LLM request timeout; the failure-line detector no longer mistakes "0 failed"/"no failures" for failures. Falls back to local classification on LLM failure. Returns a `skipped` result when no test runner is detectable; a timed-out command kills its whole process group. When the combined test output is longer than 20000 characters the TAIL is kept (prefixed with a `[... output truncated ...]` marker) instead of the head, because test runners print the pass/fail summary at the end. |
| `furrow/core/orchestrator.py` | `Orchestrator(goal, client, settings, state_store, event_bus, resume)`. Tracks tasks for real, executes in dependency waves honoring `dependencies`, bounds concurrency with `max_parallel_tasks`, honors `max_cycles` (0 = unlimited) with a no-progress guard (counter and previous plan signature persisted in run state; restored on resume so guard does not restart from zero; the state file is written before each cycle's guard evaluation, so the persisted counter lags the live one by one cycle and the persisted signature is the previous cycle's; applies to skipped/no-signal runs). The no-progress guard is evaluated BEFORE the state is saved, so the state is written exactly once per completed cycle with the CURRENT guard counter and one STATE_SAVED event. A plan merge preserves the terminal status of a task for reporting (completed stays completed, a task that failed with an exhausted retry budget stays failed, a blocked task stays blocked) and preserves its attempt counter. A task that already COMPLETED is not executed again in the next cycle when the new plan keeps the same description and files (it is skipped, no attempt consumed, a log line says so), while a task whose description or files changed is re-run. The retry budget is enforced within a cycle, not across cycles. `tasks_blocked` are counted and reported. Keeps `original_goal` intact and amends a separate working goal with test-failure feedback, persists state at the end of every completed cycle (a cycle that ends early — a planner error, or a plan with no tasks — is not persisted, so the saved cycle count can lag the reported one by one), emits events, prints console progress. On resume the persisted state is loaded exactly once; task exceptions are caught inside the task runner and converted into task failures (one attempt consumed, retried while the budget lasts, an ERROR event emitted, a console line) — only `asyncio.CancelledError` propagates. The cycle loop is fail-safe: a planner error, an execution error or a tester error emits an ERROR event, sets a stopped reason (`planner failed: ...`, `execute failed: ...`, `tester failed: ...`) and ends the loop cleanly with a normal `RunResult`; `asyncio.CancelledError` still propagates. An exception raised while handling a worker's result is converted into a task failure that consumes one attempt (it used to be silently swallowed by `asyncio.gather(..., return_exceptions=True)`), so a task is never left in an inconsistent state. A failed state write emits an ERROR event with the path that could not be written and does NOT emit STATE_SAVED. Accumulates LLM usage per cycle and per run; `tests_run` is incremented per test execution. `run() -> RunResult` (now includes `tasks_blocked`, `tests_run`, `usage`, `files_changed`), `status() -> dict`. |
| `furrow/core/events.py` | `Event`, `EventType`, `EventBus` (fan-out pub/sub). Each subscriber gets a BOUNDED queue (1000 events); `emit` never blocks and never raises. When a subscriber is not draining, the oldest queued event is dropped so the newest event is always delivered, and the number of dropped events is counted and exposed (via `dropped_count`/`dropped_for`). |
| `furrow/core/state.py` | `RunState`, `HistoryEntry`, `StateStore`. `RunState` carries `goal`/`original_goal`/`cycles`/`tasks`/`history`/`updated_at` plus `usage`, `no_progress` and `prev_sig` and the new `tests_run` field (the number of test-suite executions) in the persisted state; `HistoryEntry` carries the per-cycle fields (`cycle`, `goal`, `plan_rationale`, `task_summaries`, `tests_passed`, `test_summary`, `timestamp`) plus a per-cycle `usage`, so usage survives in the state file. Atomic writes (temp file + `os.replace`); corrupt state files are ignored. |
| `furrow/cli/main.py` | Click CLI: `furrow start`, `furrow status`, `furrow resume`, `furrow web`. The CLI releases the LLM client when a run ends, including when the run raises. Honours `FURROW_LOG_LEVEL`. Run summary includes a `Tasks blocked:` count; when no test was executed in a run, reports `Result: NOT RUN - no tests were executed.`; otherwise reports `Result: SKIPPED - no test runner detected.`, `Result: PASSED - all tests passed.`, or `Result: FAILED - tests did not pass.` as appropriate. Prints `LLM usage:` (calls/tokens) and `Files changed:` lines. The `web` command reports a friendly error and exits 1 when the server cannot start; `--max-cycles` rejects negative values at the command line; `furrow start` / `furrow resume` exit with status 1 when the run result is FAILED (tests ran and did not pass) and 0 for PASSED, SKIPPED (no test runner detected) and NOT RUN (no tests were executed). |
| `furrow/web/server.py` | FastAPI app: `GET /` (single-page UI with a status header whose bindings `goal-display`, `cycle`, `state`, `stopped-reason`, `usage` are declared so status renders correctly; run summary appended to event log only when it changes), `GET /api/state`, `GET /api/status`, `POST /api/start`, `WS /ws` streaming orchestrator events with final `{"type": "done", ...}` frame. A second concurrent run is refused, and the run registry cannot get stuck in the "running" state. The web server releases the LLM client when a run finishes, errors or is cancelled, and the task status is mapped to a known set (with an `unknown` fallback) before it is used in a CSS class name, while the status text is still written with textContent. The `.status-skipped` CSS rule styles skipped task panels (they were unstyled before); `max_cycles` is bounded with `Field(None, ge=0)` on both the `POST /api/start` body and the websocket message, and the settings built for a run are validated instead of bypassing validation. |
| `furrow/__init__.py` | Package entry point; re-exports `Settings` and `LLMClient`. |
| `furrow/agents/__init__.py` | Re-exports `PlannerAgent`, `TesterAgent`, `WorkerAgent`. |
| `furrow/agents/prompts.py` | System prompts for planner/worker/tester and the four `WORKER_TOOLS` schemas (`read_file`, `write_file`, `list_files`, `run_command`). |
| `furrow/cli/__init__.py` | Empty placeholder module. |
| `furrow/core/__init__.py` | Re-exports `Event`, `EventBus`, `EventType`, `Orchestrator`, `RunState`, `StateStore`. |
| `furrow/logging.py` | `configure_logging()` / `get_logger()`; structlog with a stdlib fallback. |
| `furrow/web/__init__.py` | Re-exports the FastAPI `app` and `run()` helper. |

## File tree

```text
furrow/
├── __init__.py
├── config.py
├── llm.py
├── agents/
│   ├── __init__.py
│   ├── parsing.py
│   ├── planner.py
│   ├── prompts.py
│   ├── tester.py
│   └── worker.py
├── cli/
│   ├── __init__.py
│   └── main.py
├── core/
│   ├── __init__.py
│   ├── events.py
│   ├── orchestrator.py
│   └── state.py
└── web/
    ├── __init__.py
    └── server.py
```

## Data flow (one cycle)

1. **Plan** — Orchestrator calls `PlannerAgent.plan(goal, context=feedback)`. Context includes test failures from the previous cycle if present.
2. **Merge** — `Orchestrator._merge_plan()` copies the new plan's description/files/dependencies onto each existing task and preserves terminal status for reporting (`completed` stays completed, a task that failed with an exhausted retry budget stays `failed`, a blocked task stays `blocked`) and preserves the attempt counter; a recorded per-task result/error is not wiped when the incoming plan carries none. A completed task with an unchanged description and files is skipped; only a task whose description or files changed is attempted again. Sets the active task set.
3. **Execute** — `_plan_waves()` computes dependency waves. Tasks are executed concurrently per wave up to `max_parallel_tasks`. For each task, `WorkerAgent.run()` runs a tool-calling loop (`read_file`/`write_file`/`list_files`/`run_command`) until no tool calls or iteration budget exhausted. Results update `TaskModel.status/result/error/attempts/timestamps`.
4. **Test** — `TesterAgent.run(original_goal, tasks)` detects the test command (or falls back to classification) and returns `TestResult`.
5. **Persist & emit** — Events are emitted throughout (cycle/task/test/state). `RunState` (goal/original_goal/cycles/tasks/history) is saved atomically to `state_path()`; a failed state write is reported as an ERROR event (with the path that could not be written) and does not emit STATE_SAVED. If tests fail, `goal` is amended with failure details for the next cycle; a no-progress guard can stop the run.

## Events and state

- **Events** (`furrow/core/events.py`): `EventType` values include `CYCLE_STARTED`, `PLAN_CREATED`, `TASK_STARTED`, `TASK_FINISHED`, `TEST_STARTED`, `TEST_FINISHED`, `STATE_SAVED`, `LOG`, `ERROR`. `EventBus` provides fan-out pub/sub. A `STATE_SAVED` event is emitted only when the state file was actually written; a failed write emits an `ERROR` event (with the path that could not be written) instead. The web UI and CLI consume these events.
- **State file**: `<workspace>/.furrow/state.json` (configurable via `FURROW_STATE_FILE`). Contains `RunState` with `goal`, `original_goal`, `cycles`, `tasks`, `history`, `updated_at`, accumulated LLM `usage`, `version` (default 1), `no_progress` (no-progress guard counter), `prev_sig` (previous plan signature) and `tests_run`. Each subscriber gets a BOUNDED queue (1000 events); when a subscriber is not draining, the oldest event is dropped so the newest is always delivered and the drop count is tracked. `RunResult` includes `passed`, `cycles`, `tasks_completed`, `tasks_failed`, `tasks_blocked`, `tests_run`, `summary`, `stopped_reason`, `skipped`, `usage`, and `files_changed`. Written atomically; corrupt files are ignored on load.

## Permissions

Kilo integration uses open permissions by default (`bash: allow`, `edit: allow`, `read: allow`) as configured in `kilo.json`.

## Version control

A Furrow run works in a disposable workspace, so a change is only durable once it is committed **and** pushed to `origin`. A local-only commit is lost when the workspace is discarded.

- Work on the branch the session was given (`kilo/<name>`). Do not rewrite history on a shared ref — no force pushes, no rebases of published branches.
- Commit at the end of every cycle that produced changes. The message describes the change, not the process.
- Push the branch to `origin` before the session ends, and set it to track the remote branch.
- If the session was given a base branch (a `session/agent_<uuid>` branch), fast-forward that branch to the run's tip and push it too, so the next run starts from the improved code rather than re-deriving it. A fast-forward is safe because the old tip stays in history.
- Leave `.furrow/` (run state) untracked — it is local scratch state, covered by `.gitignore`.

## Usage

- **Kilo**: Run `/furrow <goal>` to start the Kilo orchestrator loop.
- **CLI**: `furrow start "<goal>"`, `furrow status`, `furrow resume`, `furrow web` (default http://127.0.0.1:8000).
- **Web API/WebSocket**: `GET /api/state`, `GET /api/status`, `POST /api/start`, `WS /ws` streams events and ends with `{"type": "done", ...}`.

## Future Work

- Richer web UI with per-agent panels — delivered (status header with cycle/state/usage, per-task panels, timestamped event log).
- A richer `furrow` package plugin system
- Multi-provider model routing
- Cost accounting: token accounting and currency cost estimation are both delivered (per-call, per-cycle and per-run counts and estimated USD cost via the built-in pricing table, overridable with `FURROW_MODEL_PRICING`, persisted in state and printed by the CLI).
