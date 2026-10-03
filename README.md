# Furrow

Autonomous coding agent that runs a **plan → execute → test → repeat** loop against a
workspace until the goal is met.

Each cycle, Furrow asks a planner LLM to break the goal into independent tasks, executes the
tasks with worker agents in dependency-ordered parallel waves, runs the project's test
runner, and feeds the results back into the next plan. Cycles repeat until the goal is
complete, the cycle budget runs out, you stop it, or the loop notices it is making no
progress. There is a CLI for a single terminal, and a web console for live monitoring.

## Install

Requires **Python >= 3.10**.

```bash
pip install furrow
```

For a source checkout:

```bash
pip install -e ".[dev]"
```

## Usage

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment (or point
`FURROW_PROVIDER=ollama` at a local Ollama server). Without a key — or a reachable Ollama
endpoint — the LLM calls fail.

```bash
furrow start "Add JWT auth to the API"   # run the loop
furrow plan  "Add JWT auth to the API"   # dry run: print the plan, execute nothing
furrow web                               # live web console
furrow version                           # version and Python
```

The `furrow` entry point also runs as a module, which is handy in a source checkout:

```bash
python3 -m furrow.cli.main start "Add JWT auth to the API"
```

### `furrow start "<goal>"`

Runs the loop. `GOAL` is optional; if omitted you are prompted for one, and a blank answer
is rejected.

```bash
furrow start "Add JWT auth to the API"
furrow start "Fix the flaky retry test" --cycles 5
furrow start "Port the CLI to Typer" --provider openai --model gpt-4o --max-parallel 3
furrow start "Fix the build" --workspace ./service --test-command "make test" --test-timeout 300
```

### `furrow plan "<goal>"`

Dry run. Asks the planner for exactly one plan, prints it as a task table plus its
rationale, and exits. The workspace is never written to. Takes the same options as `start`.

```bash
furrow plan "Add JWT auth to the API"
```

### `furrow web`

Serves the web console. It binds `0.0.0.0` on port `8000` by default, so the browser URL is
<http://localhost:8000> while every interface is exposed — pass `--host 127.0.0.1` to bind
loopback only.

```bash
furrow web
furrow web --host 127.0.0.1 --port 9000
```

### `furrow version`

```bash
furrow version
# furrow 0.1.0 (python 3.10.12)
```

### Options

`start` and `plan` share these options. Only options you actually pass are overridden;
everything else keeps coming from the environment or `.env`.

| Option | Meaning |
| --- | --- |
| `--model TEXT` | LLM model to run the loop with. |
| `--provider [anthropic\|openai\|ollama]` | LLM provider. |
| `--workspace DIRECTORY` | Workspace root the agents read and write. |
| `--cycles INTEGER` | Cycle budget. **`0` means unlimited** (and `0` is the default). |
| `--max-parallel INTEGER` | Tasks executed at the same time (>= 1). |
| `--test-command TEXT` | Command the tester agent runs. |
| `--test-timeout FLOAT` | Seconds allowed for a test run (>= 0). |

`furrow web` takes `--host` (default `0.0.0.0`) and `--port` (default `8000`, range
1-65535).

### Stopping a run

Press **Ctrl+C**. The first interrupt requests a graceful stop that lands after the current
step; a second aborts immediately. The CLI exits `0` on goal-complete or a clean stop, `1`
when the loop halted without meeting the goal, and `130` when it was aborted.

## Web console

`furrow web` serves a single page console at `/` and a small JSON/websocket API.

| Endpoint | Behaviour |
| --- | --- |
| `GET /api/health` | `{"status": "ok", "version": "..."}`. |
| `GET /api/state` | Snapshot of the active run, or of the last run to finish. `{"goal", "cycles", "done", "tasks", "history"}`. |
| `POST /api/run` | Starts a run in the background. Body: `{"goal": ..., "max_cycles": ..., "model": ...}`. |
| `POST /api/stop` | Asks the active run to stop before its next cycle. |
| `WS /ws` | Live event stream. |

`POST /api/run` returns **202 Accepted** immediately — the run continues in the background.
It returns **409 Conflict** if a run is already in flight, and **422 Unprocessable Entity** if
`goal` is empty or only whitespace. `POST /api/stop` returns **404 Not Found** when nothing
is running, and stops are honoured between cycles, not mid-cycle.

**A single run is served at a time.** The `/api/run` and `/ws` paths share one global
registry slot: while a run is active, a second start is rejected with 409, and a `WS /ws`
client receives an `error` frame instead of a second orchestrator. Closing the websocket
mid-run stops the run so the loop cannot outlive its client.

The websocket expects a JSON start frame as its first message, then streams one frame per
orchestrator event, shaped exactly like an event object:

```json
{"type": "task_end", "cycle": 2, "message": "...", "data": {"task": {...}}, "timestamp": 0}
```

Event types: `cycle_start`, `plan`, `task_start`, `task_end`, `test_start`, `test_end`,
`cycle_end`, `error`, `done`.

```bash
curl -s localhost:8000/api/health
curl -s -X POST localhost:8000/api/run \
  -H 'content-type: application/json' \
  -d '{"goal": "Add JWT auth to the API", "max_cycles": 3}'
curl -s -X POST localhost:8000/api/stop
```

## Configuration

Every setting below is a `Settings` field and can be overridden with a **`FURROW_`-prefixed
environment variable** or a **`.env` file** in the working directory (uppercase the field
name and prefix it: `FURROW_MAX_PARALLEL_TASKS=8`).

| Setting | Default | Meaning |
| --- | --- | --- |
| `provider` | `anthropic` | `anthropic`, `openai`, or `ollama`. |
| `model` | `claude-sonnet-4-20250514` | Model for general loop calls. |
| `planner_model` | `claude-3-5-haiku-20241022` | Model used by the planner. |
| `worker_model` | `claude-3-5-sonnet-20241022` | Model used by workers. |
| `tester_model` | `claude-3-5-sonnet-20241022` | Model used by the tester. |
| `anthropic_api_key` | unset | Falls back to `ANTHROPIC_API_KEY`. |
| `openai_api_key` | unset | Falls back to `OPENAI_API_KEY`. |
| `ollama_base_url` | `http://localhost:11434` | Base URL of the Ollama server. |
| `max_parallel_tasks` | `5` | Concurrent workers. Set with `--max-parallel`. |
| `max_cycles` | `0` | Cycle budget; **`<= 0` means unlimited**. Set with `--cycles`. |
| `workspace` | current directory | Root the agents read and write. Set with `--workspace`. |
| `max_tokens` | `4096` | Completion cap per request. |
| `request_timeout` | `120.0` | Seconds allowed for an LLM request. |
| `llm_max_retries` | `3` | Attempts per LLM call, with exponential backoff. |
| `test_command` | unset | Single command the tester runs. Set with `--test-command`. |
| `test_timeout` | `120.0` | Seconds allowed for a test run. Set with `--test-timeout`. |
| `log_level` | `INFO` | Log level for the server process. |

**`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` must be set** (or a local Ollama endpoint must be
reachable via `FURROW_OLLAMA_BASE_URL`); there is no anonymous mode and the loop cannot
plan or execute without one.

```bash
cat > .env <<'EOF'
FURROW_PROVIDER=openai
FURROW_MODEL=gpt-4o
FURROW_MAX_PARALLEL_TASKS=8
FURROW_MAX_CYCLES=10
FURROW_WORKSPACE=/path/to/project
FURROW_TEST_COMMAND=pytest -q
EOF
```

## How the loop terminates

The loop stops for exactly four reasons:

1. **`request_stop()`** — Ctrl+C in the CLI, or the web console's Stop button
   (`POST /api/stop`). Honoured between cycles, so the current step finishes first.
2. **Cycle limit** — `max_cycles` (or `--cycles`) when positive. **`max_cycles <= 0`, which
   is the default, means unlimited.**
3. **Goal complete** — every task the planner has produced is `completed` and there are no
   outstanding test failures.
4. **No-progress guard** — after **3 consecutive cycles** that produce no new tasks and
   change no task state, the loop stops rather than spinning forever. This is what stops a
   goal that is unreachable or already satisfied.

A failing test suite does **not** end the run: the failures are folded into the next
cycle's planner prompt as a follow-up fix request, so the loop keeps going until the budget,
the stop request, the goal, or the no-progress guard says otherwise.

## Supported test runners

With no `test_command` configured, the tester tries these in order and uses the first one
that actually launches in the workspace:

| Runner | Command |
| --- | --- |
| pytest | `pytest -q` |
| pytest (module form) | `python -m pytest -q` |
| npm | `npm test -- --silent` |
| pnpm | `pnpm test -- --silent` |
| yarn | `yarn test --silent` |
| cargo | `cargo test -q` |
| go | `go test ./...` |

A workspace with no runner at all is not a failure — the tests are reported as passing with
"no test runner found". To pin a specific command, set `FURROW_TEST_COMMAND` or pass
`--test-command "your command"`; it is then split with `shlex` and run on its own. A runner
that exceeds `--test-timeout` is killed and the run is reported as failed.

## Development

```bash
pip install -e ".[dev]"
pytest
python3 -m ruff check
python3 -m mypy
```

Python **>= 3.10** is required. `ruff` enforces an explicit rule set (`[tool.ruff.lint]` in
`pyproject.toml`) and `mypy` type-checks the `furrow` package.