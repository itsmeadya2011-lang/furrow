# Furrow

Autonomous coding agent with an infinite parallel development loop.

Furrow takes a goal, breaks it into parallel tasks, runs them concurrently with an
LLM, runs the project's test suite, feeds the failures back into the next plan and
repeats until the goal is done or a termination guard fires. It ships as a click CLI
(`furrow start`), a FastAPI web UI (`furrow web`), and as Kilo subagents for the TUI.

## Install

```bash
pip install furrow
```

Python 3.10 or newer. For a checkout with the dev tooling:

```bash
pip install -e ".[dev]"
```

Configure credentials before running anything:

- `ANTHROPIC_API_KEY` for the `anthropic` provider (the default),
- `OPENAI_API_KEY` for the `openai` provider,
- nothing for `ollama` — it talks to a local server and needs no key.

Both keys are also readable as `FURROW_ANTHROPIC_API_KEY` / `FURROW_OPENAI_API_KEY`;
the unprefixed names are used as a fallback. A `.env` file in the working directory is
loaded automatically.

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow status
furrow web
furrow --version
```

### `furrow start [GOAL]`

`GOAL` is optional; if omitted you are prompted for it.

| Flag | Meaning |
| --- | --- |
| `--provider [anthropic\|openai\|ollama]` | LLM provider to use |
| `--model TEXT` | Override the default model |
| `--planner-model TEXT` | Model used by the planner |
| `--worker-model TEXT` | Model used by workers |
| `--tester-model TEXT` | Model used by the tester |
| `--max-cycles INT` | Cycle budget; `0` means unlimited |
| `--max-parallel INT` | Maximum tasks executed in parallel |
| `--workspace PATH` | Directory Furrow is allowed to edit |
| `--test-command TEXT` | Command used to run the tests |
| `--quiet` | Silence progress output |
| `--verbose` | Print verbose plan dumps |

`--quiet` wins over `--verbose`: the quiet console swallows the plan panels either way.

### `furrow web`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host TEXT` | `127.0.0.1` | Interface to bind |
| `--port INT` | `8000` | Port to bind |
| `--reload` | off | Reload on source changes (dev only) |

### `furrow status`

Prints the most recently saved run state: goal, cycles run, task status counts, tests
passed/failed, halt reason, a per-task table, and the last 5 history entries. Accepts
`--workspace PATH` to read a different workspace's state file. Prints
`No saved state at <path>` and exits 0 when there is nothing to show.

### Exit codes

`furrow start`:

- `0` — the run finished with the goal satisfied,
- `1` — the run halted without reaching the goal, or the run raised,
- `130` — interrupted with Ctrl+C.

`furrow status` exits `0` in every case, including when no state file exists.
`furrow --version` prints the installed package version.

## Configuration

Every field of `furrow.config.Settings` is read from the environment with the
`FURROW_` prefix (case-insensitive) or from a `.env` file. CLI flags are layered on
top of these values.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | Provider: `anthropic`, `openai` or `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model for all roles |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the planner |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by workers |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the tester |
| `FURROW_OLLAMA_MODEL` | `llama3.1` | Model name for the `ollama` provider |
| `FURROW_ANTHROPIC_API_KEY` | unset | Anthropic key; falls back to `ANTHROPIC_API_KEY` |
| `FURROW_OPENAI_API_KEY` | unset | OpenAI key; falls back to `OPENAI_API_KEY` |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL of the Ollama server |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Concurrency cap; also caps the plan size |
| `FURROW_MAX_CYCLES` | `0` | Cycle budget; `0` means unbounded |
| `FURROW_MAX_TASK_ATTEMPTS` | `2` | Attempts per task before it is left failed |
| `FURROW_MAX_CONSECUTIVE_ERRORS` | `3` | Consecutive erroring cycles tolerated before halting |
| `FURROW_STAGNATION_LIMIT` | `3` | Identical repeated failure sets tolerated before halting |
| `FURROW_TEST_COMMAND` | unset | Test command; auto-detected when unset |
| `FURROW_TEST_TIMEOUT` | `120` | Seconds before the test process is killed |
| `FURROW_MAX_OUTPUT_CHARS` | `8000` | Cap on captured task/test output kept in state |
| `FURROW_WORKSPACE` | current directory | Root the agent reads and edits |
| `FURROW_STATE_PATH` | `<workspace>/.furrow/state.json` | Where run state is written |
| `FURROW_VERBOSE` | `false` | Print full plan dumps |
| `FURROW_TEMPERATURE` | `0.0` | Sampling temperature sent to the provider |
| `FURROW_MAX_TOKENS` | `4096` | Max tokens per completion |
| `FURROW_LOG_LEVEL` | `INFO` | Declared log level; no logging setup reads it today |

When `FURROW_TEST_COMMAND` is unset, the tester picks a runner from marker files in the
workspace: `python -m pytest -q` for a Python project, `npm test -- --silent` for a
`package.json` with a `test` script, `cargo test -q` for `Cargo.toml`, `go test ./...`
for `go.mod`. With no marker file the suite is skipped and reported as passing.

## How the loop works

Each cycle (`furrow/core/orchestrator.py`):

1. **Plan** — the planner turns the goal plus the current context into 1-5 independent
   tasks. The plan is normalized: at most `max_parallel_tasks` tasks, sequential ids,
   dependency edges that are known and acyclic.
2. **Execute** — every task in the plan runs concurrently under an
   `asyncio.Semaphore` sized by `max_parallel_tasks`. Each worker gets the task
   description, the files it owns and their current contents. Failures are collected,
   then failed tasks get one more attempt if they still have budget.
3. **Test** — the configured (or detected) test command runs in the workspace with
   `test_timeout` enforced. The exit code is authoritative: `0` passes, anything else
   (including a timeout) fails. The model is asked only to summarize and extract
   failure lines.
4. **Feed back** — failures are de-duplicated into the accumulated fix instructions
   (capped at 4000 characters), which are handed to the next cycle's planner and
   tester. The goal itself is never overwritten, so the loop cannot drift into
   chasing a different objective.
5. **Persist and report** — run state is saved, and a `cycle_end` event is emitted with
   the task statuses, pass/fail and error counters.

Progress is reported on an in-process event bus (`furrow/core/events.py`) with the
kinds `run_start`, `cycle_start`, `plan`, `task_start`, `task_end`, `test_result`,
`cycle_end`, `run_end`, `error` and `notice`. The CLI renders them as Rich panels and
the web server forwards them to the browser.

### Termination conditions

The default cycle budget is `0`, i.e. unbounded, so the other guards are what keep a
run finite and honest:

1. **Goal satisfied** — the tests passed *and* every task in the plan completed. The
   planner returning no tasks while nothing is left to fix also ends the run cleanly.
2. **`max_cycles` reached** — only applies when `max_cycles > 0`.
3. **Stagnation** — the same test-failure set repeated `stagnation_limit` times in a
   row (compared by a hash of the sorted failure lines). Without this the loop burns
   tokens re-planning the same broken fix.
4. **Consecutive-error budget** — `max_consecutive_errors` cycles in a row containing a
   planner, tester or worker exception. This is what catches a misconfigured provider
   or missing API key instead of retrying forever.
5. **Manual stop** — a cooperative `stop()` call, used by the web UI's stop button, plus
   Ctrl+C in the CLI (exit code 130).

A run ends when one of those five fires, and its exit status distinguishes "done" from
"gave up". Termination is not unconditional, though: `0` disables the `max_cycles`,
`stagnation_limit` and `max_consecutive_errors` guards, and even at the defaults a cycle
that keeps completing its tasks while producing a different failure set every time never
repeats a signature and never raises, so the loop keeps going.

## State persistence

State is written after every cycle and again at the end of the run to
`<workspace>/.furrow/state.json` (override with `--workspace` or `FURROW_STATE_PATH`).
Writes are atomic: a temp file in the same directory plus `os.replace`. A missing or
corrupt file is reported as "no state" rather than an error, so a broken state file
can never stop the loop.

The document contains the `goal`, the accumulated `fix_instructions`, the `cycles`
count, the `tasks` with their status, attempts and results, the `done` flag, the
`halted_reason`, and a per-cycle `history` (pass/fail, summary, failures, task
statuses). `furrow status` reads exactly this file.

## Web UI

```bash
furrow web
```

Then open the URL uvicorn prints — `http://127.0.0.1:8000/` by default. The default
host is loopback on purpose: a run executes model-written code in the workspace, so
binding every interface has to be an explicit `--host` choice.

| Route | Purpose |
| --- | --- |
| `GET /` | The single-page UI (no CDN, inline styles and script) |
| `GET /api/health` | `{"status": "ok"}` |
| `POST /api/runs` | Start a run from `{"goal", "model"?, "max_cycles"?}`; `202` with `{run_id, state}`, `409` if a run is already active |
| `GET /api/runs` | Recent runs, newest first |
| `GET /api/runs/{run_id}` | Current `RunState` for a run; `404` if unknown |
| `POST /api/runs/{run_id}/stop` | Ask a run to stop: `{"stopped": bool}`, `404` if unknown |
| `WS /ws?run_id=...` | Stream one run's events, then its final state, then close |

The page starts a run with `POST /api/runs` and then opens `WS /ws?run_id=<id>`, which
streams one JSON frame per event — `{"kind", "cycle", "message", "data"}` — followed by
a final `{"kind": "state", "run_id", "state"}` frame before the socket closes. Only one
run is in flight at a time. (A socket opened without `run_id` can start a run instead by
sending `{"goal": ...}` as its first frame.) Agent output is only ever inserted into the
page as `textContent`, never as markup.

## Development

```bash
python -m pytest
ruff check .
mypy
```

All three are configured in `pyproject.toml` (`[tool.pytest.ini_options]` with
`asyncio_mode = "auto"` and `testpaths = ["tests"]`, `[tool.ruff]` with a 100-column
line length and the `E, F, I, UP, B, ASYNC` rule set, and `[tool.mypy]` over the
`furrow` package with `disallow_untyped_defs`). All three are expected to be clean.
