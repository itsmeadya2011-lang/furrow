# AGENTS.md

Guidance for AI coding agents working in the Furrow repository.

## 1. Project overview

Furrow is an autonomous coding agent that plans, implements, and tests changes to a codebase through an iterative loop. It coordinates a planner, a worker (which edits files using real tool calls), and a tester to move toward a goal while persisting state and optionally streaming progress via a web UI.

Key entry points:
- `furrow start "<goal>"` — start or resume a run
- `furrow status` — show persisted state of the last run for a workspace
- `furrow resume` — continue a persisted run
- `furrow web` — serve the web UI and stream orchestrator events (default http://127.0.0.1:8000)

## 2. Build / development commands

Install (editable, with dev deps):

```bash
pip install -e ".[dev]"
```

Run the offline test suite (no API keys required):

```bash
python -m pytest -q
```

Lint:

```bash
python -m ruff check .
```

Type check:

```bash
python -m mypy furrow
```

The CLI script is `furrow = "furrow.cli.main:main"`.

## 3. Key architecture

The loop runs cycles of four stages:

1. **Plan** — `PlannerAgent.plan(goal, context)` creates a `Plan` with tasks (id, description, files, dependencies).
2. **Execute** — `Orchestrator` computes dependency waves and runs tasks concurrently (bounded by `max_parallel_tasks`). Each task runs `WorkerAgent.run()`, a real tool-calling loop with `read_file`, `write_file`, `list_files`, `run_command`. Iteration budget is `max_worker_iterations`; exhausting it is a failure.
3. **Test** — `TesterAgent.run()` detects the test command (pytest/npm/pnpm/yarn/cargo/go) and returns a `TestResult`. Exit status is ground truth.
4. **Persist & emit** — `RunState` is saved atomically to `<workspace>/.furrow/state.json`. On test failure, the working goal is amended with failure details for the next cycle. `max_cycles` controls termination (0 = unlimited).

Fail-safe: planner, execution, tester, or state-save errors emit an `ERROR` event instead of crashing the run; only `asyncio.CancelledError` propagates.

## 4. Important conventions

### State persistence is atomic

State writes use temp file + `os.fsync` + `os.replace` (`furrow/core/state.py`). Corrupt state files are ignored and the run starts fresh. A cycle that ends early (planner error, or plan with no tasks) is not persisted, so saved cycle count can lag the reported cycle count by one.

### Event bus is bounded

Each subscriber gets a bounded queue of 1000 events (`furrow/core/events.py`). `emit` never blocks and never raises. When a subscriber is not draining, the oldest queued event is dropped so the newest is always delivered; dropped count is tracked via `dropped_count`/`dropped_for`. Event types include `CYCLE_STARTED`, `PLAN_CREATED`, `TASK_STARTED`, `TASK_FINISHED`, `TEST_STARTED`, `TEST_FINISHED`, `STATE_SAVED`, `LOG`, `ERROR`.

### Commands run via `/bin/sh -c` with hardened guards

Worker commands run via `/bin/sh -c` in their own process group (timeout kills the whole tree). Commands are normalized, split into segments on `;`, `&&`, `||`, `|`, newlines, and every segment and command substitution (`$(...)`, backticks) is checked. Denied: destructive programs by name (`mkfs*`, `dd`, `fdisk`, `parted`, `shutdown`, `reboot`, `poweroff`, `halt`, `init`), destructive patterns (`rm -rf /` variants, `rm -fr /`, `rm -rf /*`, `chmod -R 777 /`, fork bomb, `> /dev/sda`), force pushes (`git push --force`, `-f`, `--force-with-lease`), and redirections targeting outside the workspace (absolute paths, `..`, `~`, `$HOME`). Workspace redirections are allowed.

### LLM provider clients are lazy and released via `aclose()`

Provider clients (Anthropic, OpenAI, Ollama) are created lazily and released when a run ends: `LLMClient.aclose()` closes the ones that were created and leaves the client usable afterwards. Tenacity retry on transient errors (408/409/425/429/5xx); non-retryable on 4xx. Malformed provider responses raise a clear `ValueError`.

### Settings use `FURROW_` prefix with pydantic-settings

`Settings` reads env vars with the `FURROW_` prefix and a `.env` file if present (`furrow/config.py`). Variables include `FURROW_PROVIDER`, `FURROW_MODEL`, `FURROW_PLANNER_MODEL`, `FURROW_WORKER_MODEL`, `FURROW_TESTER_MODEL`, `FURROW_*_API_KEY`, `FURROW_OLLAMA_BASE_URL`, `FURROW_MAX_PARALLEL_TASKS`, `FURROW_MAX_CYCLES`, `FURROW_MAX_WORKER_ITERATIONS`, `FURROW_MAX_RETRIES`, `FURROW_MAX_TASK_RETRIES`, `FURROW_LOG_LEVEL`, `FURROW_REQUEST_TIMEOUT`, `FURROW_MAX_TOKENS`, `FURROW_STATE_FILE`, `FURROW_WORKSPACE`. Numeric settings are validated (bounds enforced when settings are built).

### Module-level `settings` singleton must not be mutated

Do not mutate the module-level `settings` singleton. Settings are built per-run and validated; mutation would leak across cycles/runs.

### Tool arguments are untrusted input

Tool argument values that are not objects return an `error: invalid tool arguments` tool result instead of raising and failing the whole task. Paths are sandboxed to the workspace. Read output longer than the limit is tail-truncated with a `[... output truncated ...]` marker.

## 5. Testing conventions

Tests run offline and require no API keys (`tests/` directory, `asyncio_mode = "auto"`). Use fakes/mocks for LLM providers and external services; never hit real endpoints. Place new tests under `tests/`.

## 6. Version control guidance

Furrow runs in a disposable workspace, so a change is only durable once committed **and** pushed to `origin`. A local-only commit is lost when the workspace is discarded.

- Work on the branch the session was given (`kilo/<name>`). Do not rewrite history on a shared ref — no force pushes, no rebases of published branches.
- Commit at the end of every cycle that produced changes. The message describes the change, not the process.
- Push the branch to `origin` before the session ends, and set it to track the remote branch.
- If given a base branch (`session/agent_<uuid>`), fast-forward that branch to the run's tip and push it too, so the next run starts from improved code.
- Leave `.furrow/` (run state) untracked — it is local scratch state, covered by `.gitignore`.
