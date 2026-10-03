# Furrow

Autonomous coding agent with an unbounded parallel development loop.

Give it a goal. It plans the next slice of work, runs those tasks in parallel, verifies the result with the project's own tests and linters, and re-plans against whatever failed — until the goal is proved or a guard trips.

## Install

`pip install furrow` does **not** install this code: the `furrow` name on PyPI belongs to an unrelated project. Install from a checkout instead (Python 3.10+):

```bash
pip install -e .          # editable, for development
pip install .             # regular install
pip install -e ".[dev]"   # plus pytest, ruff and mypy
```

## Providers

`anthropic` (default), `openai` and `ollama`.

```bash
export ANTHROPIC_API_KEY=...   # or FURROW_ANTHROPIC_API_KEY, or a .env file
export OPENAI_API_KEY=...      # or FURROW_OPENAI_API_KEY
```

Ollama needs no credentials; point at it with `FURROW_OLLAMA_BASE_URL` (default `http://localhost:11434`).

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow web                      # then open http://127.0.0.1:8000
furrow --version
furrow start --help
```

`furrow start` options — each overrides the matching setting for that one run:

| Option | Meaning |
| --- | --- |
| `GOAL` | The objective. Optional; you are prompted if it is omitted. |
| `--provider` | `anthropic`, `openai` or `ollama`. |
| `--model` | Model for every agent with no role-specific override. |
| `--planner-model` | Model that writes the plans. |
| `--worker-model` | Model that implements the tasks. |
| `--tester-model` | Model that judges the test output. |
| `--max-cycles` | Cycle budget. `0` (the default) means run until the goal is met or a guard trips. |
| `--max-parallel-tasks` | Tasks of a cycle allowed to run at once. `0` means no limit. Default 5. |
| `--workspace` | Repository the agents read and write. Default: the current directory. |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. |

`furrow web` options:

| Option | Default |
| --- | --- |
| `--host` | `127.0.0.1` |
| `--port` | `8000` |

The web UI binds loopback only and has no authentication. Binding any other interface (`--host 0.0.0.0`) publishes an autonomous agent that reads and runs in your workspace to that network; the server warns on startup when you do.

A browser session is always bounded, unlike the CLI: it runs the cycle count the page asks for, clamped to 1-50, otherwise `FURROW_MAX_CYCLES` when that is non-zero, otherwise 3. Each websocket carries one run, streams its events as they happen, and the page's Stop button cancels it.

## How the loop works

One cycle is plan → execute → verify → refine.

1. **Plan** (`furrow/agents/planner.py`). The planner sees the original goal, the current focus, and a bounded listing of workspace paths (up to 60). Its reply is parsed tolerantly: a fenced code block or surrounding prose is stripped, and every balanced `{...}`/`[...]` region is tried in turn, preferring the first plan-shaped payload so a decoy example earlier in the reply cannot win. Invalid JSON is never silently repaired — the planner gets one repair round-trip, then the cycle fails. A plan holds at most 5 tasks, clamped to `max(1, min(5, max_parallel_tasks))`, so even `max_parallel_tasks = 0` (no runtime cap) yields a one-task plan; ids are made unique and non-empty, unknown or self-referential `dependencies` are dropped, and dependency cycles are broken before the plan is accepted.
2. **Execute** (`furrow/agents/worker.py`). Tasks run in dependency waves: a task starts only once every task it depends on has completed, and it is handed those dependencies' results. Everything runnable in a wave runs concurrently as in-process coroutines (`asyncio.gather` over `WorkerAgent`), capped by an `asyncio.Semaphore` of `max_parallel_tasks` (`0` means uncapped). Tasks caught in a dependency cycle are skipped before execution, as is any task whose dependency failed or was skipped; a raising task is marked failed without aborting its siblings. Each worker is grounded in the current contents of the files it targets (up to 12 files, whole prompt capped at 40,000 characters). Workers write: `WorkerAgent` sends one grounded prompt and the reply is a JSON envelope whose `files` entries are validated and then written to the workspace (`created`/`modified`, with a backup of anything overwritten), bounded by `max_files_per_task`, `max_write_bytes` and `max_total_write_bytes`. A reply with no envelope, or a prose-only reply, is kept as the task result and changes nothing.
3. **Verify** (`furrow/agents/tester.py`). The tester detects the project's own runners (pytest, pnpm, yarn, npm, cargo, go — matched on marker files and what is on `PATH`) and also runs `ruff check .` and `mypy .` when they are installed. Each command gets `test_timeout` seconds inside a shared phase budget, children run with a scrubbed environment, and output is truncated before it reaches the model. Exit codes are authoritative: any non-zero exit, spawn error or timeout forces `passed=false` even when the model claims success, while pytest's exit code 5 (nothing collected) is not a failure. If no runner exists at all, the result is a vacuous pass (`summary` starts with `[no-test-runner]`, `TesterAgent.last_kind == "no_test_runner"`) — unverified, never a fake failure. The model then only judges the evidence it was handed. **The tester fixes nothing**: it returns `passed`, `summary` and `failures`, and fixing is the next cycle's job.
4. **Refine.** On a failing verdict the focus becomes `Fix failing tests:` plus the failure list, while the original goal stays fixed. Three identical failure fingerprints in a row trip the no-progress guard, and three consecutive planner failures stop the run.
5. **Repeat** until the goal is proved, `max_cycles` is reached (`max_cycles = 0` means unlimited), the planner returns no tasks, or a guard trips; the reason lands in `stop_reason`. Completion is strict: the most recent verification must have passed **and** every task in the final plan must be completed. "All tasks completed" alone is not success, and the loop does not exit after a single cycle.

## Configuration

Every setting below is read from a `FURROW_*` environment variable and from a `.env` file in the working directory; explicit CLI flags win over both. The `furrow.config.settings` singleton is never mutated by a run.

| Setting | Env var | Default |
| --- | --- | --- |
| `provider` | `FURROW_PROVIDER` | `anthropic` |
| `model` | `FURROW_MODEL` | `claude-sonnet-4-20250514` |
| `planner_model` | `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` |
| `worker_model` | `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` |
| `tester_model` | `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` |
| `anthropic_api_key` | `FURROW_ANTHROPIC_API_KEY` | unset (`ANTHROPIC_API_KEY` is also read) |
| `openai_api_key` | `FURROW_OPENAI_API_KEY` | unset (`OPENAI_API_KEY` is also read) |
| `ollama_base_url` | `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` |
| `max_parallel_tasks` | `FURROW_MAX_PARALLEL_TASKS` | `5` (`0` = no limit) |
| `max_cycles` | `FURROW_MAX_CYCLES` | `0` (unlimited) |
| `max_stagnant_cycles` | `FURROW_MAX_STAGNANT_CYCLES` | `3` |
| `stagnation_threshold` | `FURROW_STAGNATION_THRESHOLD` | `0.8` |
| `workspace` | `FURROW_WORKSPACE` | current directory |
| `log_level` | `FURROW_LOG_LEVEL` | `INFO` |
| `max_tokens` | `FURROW_MAX_TOKENS` | `4096` |
| `temperature` | `FURROW_TEMPERATURE` | `0.2` |
| `request_timeout` | `FURROW_REQUEST_TIMEOUT` | `120.0` seconds |
| `max_retries` | `FURROW_MAX_RETRIES` | `2` (attempts = retries + 1) |
| `max_retry_wait` | `FURROW_MAX_RETRY_WAIT` | `20.0` seconds |
| `test_timeout` | `FURROW_TEST_TIMEOUT` | `120` seconds per command |
| `apply_changes` | `FURROW_APPLY_CHANGES` | `true` (`0` = advisory mode: workers report the edits they would make and write nothing) |
| `backup_dir` | `FURROW_BACKUP_DIR` | `.furrow/backups` (relative paths are resolved inside the workspace) |
| `max_files_per_task` | `FURROW_MAX_FILES_PER_TASK` | `10` |
| `max_write_bytes` | `FURROW_MAX_WRITE_BYTES` | `200000` |
| `max_total_write_bytes` | `FURROW_MAX_TOTAL_WRITE_BYTES` | `1000000` |

A worker writes each file it was asked to change, but only after the path is checked to be a workspace-relative path that does not escape the workspace and does not land in `.git/`, `.venv/`, `node_modules/`, `__pycache__/`, `*.egg-info/`, `dist/` or `build/`; an existing file is copied to `<backup_dir>/<UTC timestamp>-<task id>/<relative path>` first and the write is refused outright if that backup cannot be made, and every file is replaced atomically. Rejected edits are listed in the task result, so the next cycle sees them. Set `FURROW_APPLY_CHANGES=0` to run advisory-only.

## The Kilo layer

The package above is standalone. `.kilo/` holds a second implementation of the same loop as Kilo agent definitions — `furrow` (orchestrator), `furrow-planner`, `furrow-worker`, `furrow-tester` — plus the `/furrow` slash command, configured by `kilo.json` at the repository root (`default_agent: furrow`; `bash`, `edit` and `read` allowed). `/furrow <goal>` runs that loop inside a Kilo session, where workers are real subagents spawned with the Task tool and do edit files, and where the user can interject mid-loop. See `.kilo/FURROW.md`.