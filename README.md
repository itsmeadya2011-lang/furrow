# Furrow

An autonomous coding agent that runs a plan/execute/test loop until your goal is met:
a **planner** turns the goal into tasks, **workers** implement those tasks in parallel, a
**tester** runs the test suite and judges the result, and the loop repeats until the goal is
done, the cycle budget runs out, or something fails.

Furrow rewrites files in your workspace and needs an LLM provider API key. It defaults to
Anthropic; OpenAI and a local Ollama server are also supported.

## Install

```bash
pip install furrow
```

For a checkout, with the development tools (pytest, ruff, mypy):

```bash
pip install -e ".[dev]"
```

Requires Python 3.10 or newer.

## Quickstart

```bash
export ANTHROPIC_API_KEY=sk-ant-...
furrow start "Add JWT auth to the API"
```

Run it with no goal and Furrow prompts for one:

```bash
$ furrow start
Enter your goal for Furrow: Add JWT auth to the API
```

Explore the interface:

```bash
furrow --help          # commands: start, web
furrow start --help    # every option, see the CLI reference below
furrow web --help
```

Stop a run with Ctrl-C; the loop halts gracefully and pending work is cancelled.

## How the loop works

Each cycle in [`furrow/core/orchestrator.py`](furrow/core/orchestrator.py) does four things:

1. **Focus.** The planner is asked for a plan. From cycle two on, the prompt is the original
   goal plus a bounded digest of what earlier cycles already completed and which test failures
   are still outstanding, so cycles build on each other instead of restarting.
2. **Plan.** The planner returns 1-5 tasks, each with an `id`, a `description`, the `files` it
   expects to touch and the task `dependencies` it needs. Unknown dependency ids are dropped,
   duplicate ids are suffixed, and a plan larger than 5 tasks is truncated.
3. **Execute.** Tasks are grouped into *dependency waves* — a task becomes runnable once all of
   its dependencies sit in an earlier wave. Tasks inside a wave run concurrently, capped by
   `max_parallel_tasks` (default `5`). A task whose dependency failed, was skipped or never ran
   is marked `skipped` and is not run at all. A dependency cycle is broken by releasing the
   remaining tasks together rather than deadlocking.
4. **Test.** The tester runs the project's test command as a subprocess in the workspace and
   asks the model whether the output means the goal is met.

Failing tests, or a plan that finished with tasks still incomplete, start another cycle. The
original goal is never rewritten; the "focus" for the next cycle is derived from it.

Workers only write files. They do not run commands, so the test command is the only code from
your project that Furrow executes.

### When the loop stops

The loop halts on exactly these conditions:

| Condition | Result |
| --- | --- |
| **Goal met** — every task of the current plan completed *and* the tester reported success, or the planner proposed no further work while the tests were already green | `run()` returns `True` |
| **Cycle budget reached** — `max_cycles > 0` and that many cycles have run | returns `False`, "Reached max_cycles limit (N). Halting." |
| **Stop requested** — `stop()` was called (web `cancel`, Ctrl-C) | returns `False`; the flag is checked at every phase boundary and before each worker starts, so no new work is started |
| **Planner failure** — the planner raised, or returned nothing usable | returns `False`, "Planner failed: … Halting." |
| **Cancellation** — `asyncio.CancelledError` | cleaned up and re-raised |

A failing tester is *not* a halt: it becomes a failed test result that feeds the next cycle. So
is a tester error, which is reported as a failing test result rather than killing the run.

> **`max_cycles = 0` means unlimited, and it is the default.**
> `furrow start` will keep spending tokens, planning and testing until it believes the goal is
> met or you interrupt it. There is no other backstop. Set a budget whenever you are not
> watching: `furrow start "…" --max-cycles 3`, or `FURROW_MAX_CYCLES=3`.

## Safety

Furrow writes files. Treat it like a junior engineer with shell access to your workspace.

- **Workers write whole files, not diffs.** Each task returns
  `{"path": "...", "content": "<complete new file>"}` entries, and every returned file is
  written verbatim — parent directories are created, existing files are overwritten. There is
  no diff application and no backup.
- **Path escapes are refused.** A path is resolved (including symlinks) against the workspace
  and must land inside it; anything reaching outside is rejected, logged as
  `worker_path_escape_rejected` and reported back as `Rejected (path escape)`.
- **Workers cannot run commands.** They read the files a task names (truncated at 10 000
  characters each) and write files.
- **The tester runs your real test command.** `TesterAgent` spawns it as a subprocess in its own
  process group, so your test code executes with your permissions and can do anything your user
  can. On timeout (`test_timeout`, default 120s) the whole process group gets `SIGTERM`, then
  `SIGKILL` after a 5 second grace period.
- **Run it somewhere you are willing to have modified.** Use a clean git checkout or a copy.
  `--workspace DIR` (or `FURROW_WORKSPACE`) points the agent at a specific directory;
  the default is the current working directory.
- **The web server is a remote-code-adjacent surface.** It binds `127.0.0.1` by default for
  this reason.

## CLI reference

Every option below defaults to *unset*, meaning "keep whatever `FURROW_*`, `.env` or the
built-in default provides". Only options you actually pass are overridden, and they are all
validated through `Settings`, so an invalid value is a usage error rather than a traceback.

### `furrow start [GOAL]`

| Option | Default | Meaning |
| --- | --- | --- |
| `GOAL` (positional) | prompted | The goal to work on. Optional: you are asked interactively when it is omitted or blank. |
| `--model NAME` | unset | Model for *every* agent. |
| `--planner-model NAME` | unset | Model for the planner only. |
| `--worker-model NAME` | unset | Model for the workers only. |
| `--tester-model NAME` | unset | Model for the tester only. |
| `--provider NAME` | unset | LLM backend: `anthropic`, `openai` or `ollama`. |
| `--max-parallel-tasks N` | unset | Tasks run concurrently inside a wave (≥ 1). |
| `--max-cycles N` | unset | Cycle budget; `0` is unlimited. |
| `--workspace DIR` | unset | Directory Furrow reads and writes. Rejected if it points at a file; a missing directory is created. |
| `--test-command CMD` | unset | Explicit shell test command; auto-detected when unset. |
| `--log-level LEVEL` | unset | Minimum log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (case-insensitive). |

**Model precedence:** `--planner-model`/`--worker-model`/`--tester-model` each set their own
field, and `--model` is applied last and overrides all four. `--model` therefore always wins,
including over the role-specific defaults.

```bash
furrow start "Add JWT auth to the API" \
  --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --max-parallel-tasks 3 \
  --max-cycles 5 \
  --workspace ./my-project \
  --test-command "pytest -q" \
  --log-level DEBUG
```

### `furrow web`

| Option | Default | Meaning |
| --- | --- | --- |
| `--host ADDR` | `127.0.0.1` | Interface to bind. The default keeps runs private to this machine; `0.0.0.0` exposes an agent that can write files to the network. |
| `--port PORT` | `8000` | TCP port (1-65535). |
| `--log-level LEVEL` | unset | Minimum log level. |
| `--max-cycles N` | unset | Cycle budget for runs started in the browser. `0` keeps the web default of 8; values above the hard cap of 50 are clamped. |
| `--max-concurrent-runs N` | unset | Runs served at the same time (≥ 1). |

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | The goal was completed. |
| `1` | The run halted without meeting the goal: stop requested, cycle budget exhausted, planner error, or no goal given at the prompt. |
| `130` | Interrupted with Ctrl-C. The run was stopped and its tasks cancelled. |
| `2` | Invalid usage: an unknown option or an out-of-range option value. |

`--help` and `--version` exit `0`.

## Web UI

```bash
furrow web                     # http://127.0.0.1:8000
furrow web --host 0.0.0.0 --port 9000 --max-concurrent-runs 4
```

The page at `/` starts one run per websocket connection and streams the loop's progress back.
The server binds **loopback only** by default. Exposing it is a deliberate step:
`--host 0.0.0.0` publishes an endpoint that can start an agent writing files in your
workspace, and the CLI prints a warning if you do it without `FURROW_WEB_TOKEN`.

### HTTP endpoints

| Method and path | Purpose |
| --- | --- |
| `GET /` | The single-page UI. With `FURROW_WEB_TOKEN` set it needs `?token=…` or the `X-Furrow-Token` header, and answers 401 otherwise. |
| `GET /health` | Liveness probe: status, version, active runs, the run cap, the cycle budget and whether auth is required. Always open and free of run data. |
| `GET /api/state` | Read-only JSON summary of the most recent run (goal, cycles, current plan, plan history, last test result, outcome). In-process only — a multi-worker uvicorn invocation would keep one summary per worker. |
| `WS /ws` | One run per connection, streaming JSON frames. |

### WebSocket protocol

Connect to `/ws`. The server accepts the socket, reads **exactly one** `start` message, and
ignores any later `start` on the same connection.

Client to server, once per connection:

```json
{"type": "start", "goal": "Add JWT auth to the API", "model": "claude-sonnet-4-20250514", "max_cycles": 8, "token": "optional"}
```

- `type` may be omitted: `{"goal": "…"}` is accepted as a `start`.
- `goal` is required, non-empty, and at most 4000 characters.
- `model` is optional (at most 200 characters) and applies to every agent for that run, like
  the CLI `--model`.
- `max_cycles` is optional, an integer from 1 to 50, and can only *lower* the server's budget.
- `token` is required only when `FURROW_WEB_TOKEN` is set.
- Binary frames, invalid JSON, non-object payloads, a missing or oversized goal and unknown
  message types are all refused with an `error` frame.

Client to server while the run is in flight:

```json
{"type": "cancel"}
```

This asks the orchestrator to stop cooperatively; the current step finishes first.

Server to client — every frame is a JSON object with a string `event` and a string `message`,
sent as it happens; the `accepted` and final `done` frames add an object `data`:

```json
{"event": "accepted", "message": "Run 3f9c1a7d2b04 accepted; at most 8 cycle(s).", "data": {"run_id": "3f9c1a7d2b04", "max_cycles": 8}}
{"event": "run", "message": "Starting Furrow on goal: Add JWT auth to the API"}
{"event": "cycle", "message": "Cycle 1 of 8 started."}
{"event": "plan", "message": "Plan for cycle 1: add a token module and wire it into the API"}
{"event": "task", "message": "Task 1 completed: Written: src/auth.py | added JWT helpers"}
{"event": "test", "message": "Tests passed: 42 passed"}
{"event": "finished", "message": "Goal complete."}
{"event": "done", "message": "Goal complete.", "data": {"done": true, "cycles": 2, "error": null}}
```

- Events on the wire are `run`, `plan`, `cycle`, `task`, `test`, `error`, `finished` and `done`.
  The first six come from the loop; `finished` is the loop's own terminal event, relayed under a
  name of its own; `done` is the session summary the web layer adds.
- The web layer also adds `accepted` (run accepted; `data` carries the run id and the cycle
  budget this run was actually given) and `warn` (the client is not reading fast enough and
  frames were dropped — the in-flight queue holds 512 frames and drops the oldest first).
- `finished` is the loop's own terminal event, emitted once per run and carrying **no** `data`;
  its `message` says why the loop stopped — goal met, cycle budget reached, stop requested,
  planner failure or cancellation. It is always followed by `done`.
- `done` is the summary: sent **exactly once** per connection, always **last**, always with
  `data`. A client may therefore act the moment it sees `done` — the payload is already on that
  frame and no second `done` can follow it.
- `data` is present on two frames only: `accepted`, with `run_id` and `max_cycles`, and the
  `done` summary, with `done`, `cycles` and `error`.
- The socket is closed with code `1000` when the run ends, on rejection, and on cancel. Another
  run needs another connection.
- Closing the connection stops the run; if it does not unwind within 10 seconds the run task is
  cancelled.

### Cycle budget

A run started from a browser is always bounded. The budget is `max_cycles` from the settings
(or `--max-cycles`) when that is positive, otherwise `8`, clamped to `50`; a `max_cycles` in
the `start` message can only lower it further. The reason is that `0` — the CLI default —
means unlimited, and one unauthenticated socket should not be able to spend API money forever.
The web layer never passes an unlimited budget to the orchestrator. The budget that actually
applies is reported as `data.max_cycles` on the `accepted` frame, so a client never has to
parse it out of the message text.

### Concurrent runs

At most `2` runs execute at once by default. A connection that arrives while every slot is
taken is sent an `error` frame and closed. Each run can itself spawn `max_parallel_tasks`
workers, so raise both knobs deliberately. The cap is `--max-concurrent-runs`, else
`FURROW_WEB_MAX_CONCURRENCY`, else `max_concurrent_runs` (`FURROW_MAX_CONCURRENT_RUNS`, `2` by
default). A `FURROW_WEB_MAX_CONCURRENCY` that is not an integer of at least 1 is logged and
ignored, falling through to `max_concurrent_runs`.

### Authentication

Set `FURROW_WEB_TOKEN` to require a token for every run:

```bash
export FURROW_WEB_TOKEN=$(openssl rand -hex 32)
furrow web --host 0.0.0.0
```

With it set, `GET /` needs `?token=…` or the `X-Furrow-Token` header, and every `start` message
must carry a matching `token`. The comparison is constant-time. `GET /health` stays open for
probes; `GET /api/state` is not token-gated, so it exposes the goal and outcome of the last run
to anyone who can reach the port — factor that into your bind decision.

## Configuration

Every setting is read from the environment with the `FURROW_` prefix, and from a `.env` file in
the working directory if one exists.

| Setting | Environment variable | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | `FURROW_PROVIDER` | `anthropic` | LLM backend for completions: `anthropic`, `openai` or `ollama`. |
| `model` | `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model id when no agent-specific override is given. Every agent passes its own role model, so this is mostly a fallback; the CLI `--model` flag sets all four model fields at once. |
| `planner_model` | `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the planner to turn a goal into a plan. |
| `worker_model` | `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by workers to implement individual tasks. |
| `tester_model` | `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the tester to interpret a test run. |
| `anthropic_api_key` | `FURROW_ANTHROPIC_API_KEY` | unset | Anthropic API key; falls back to `ANTHROPIC_API_KEY`. |
| `openai_api_key` | `FURROW_OPENAI_API_KEY` | unset | OpenAI API key; falls back to `OPENAI_API_KEY`. |
| `ollama_base_url` | `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL of the Ollama server; `/api/chat` is appended. No key required. |
| `max_parallel_tasks` | `FURROW_MAX_PARALLEL_TASKS` | `5` | Maximum tasks executed concurrently within a wave (≥ 1). |
| `max_concurrent_runs` | `FURROW_MAX_CONCURRENT_RUNS` | `2` | Maximum whole Furrow runs executing concurrently (≥ 1). The web layer falls back to this field unless `FURROW_WEB_MAX_CONCURRENCY` or `--max-concurrent-runs` overrides it. |
| `max_cycles` | `FURROW_MAX_CYCLES` | `0` | Maximum number of development cycles; `0` means unlimited. |
| `workspace` | `FURROW_WORKSPACE` | current directory | Root directory Furrow reads and writes files in; workers may not write outside it. |
| `log_level` | `FURROW_LOG_LEVEL` | `INFO` | Minimum level that gets logged: `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. |
| `test_command` | `FURROW_TEST_COMMAND` | unset | Explicit shell-style test command; when unset the tester auto-detects one. |
| `test_timeout` | `FURROW_TEST_TIMEOUT` | `120` | Seconds a single test run may take before its process group is killed (≥ 1). |
| `retry_attempts` | `FURROW_RETRY_ATTEMPTS` | `3` | Total LLM attempts for transient provider failures; `1` disables retries. |
| `retry_min_wait` | `FURROW_RETRY_MIN_WAIT` | `1.0` | Base seconds for the exponential backoff between LLM retries; waits are capped at 60s. |
| `llm_max_tokens` | `FURROW_LLM_MAX_TOKENS` | `4096` | Maximum tokens requested per LLM completion. |
| `llm_temperature` | `FURROW_LLM_TEMPERATURE` | `0.0` | Sampling temperature sent with every LLM completion. |

The web layer reads two more variables straight from the process environment, not through
`Settings`:

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `FURROW_WEB_MAX_CONCURRENCY` | `2` | Runs served at the same time (≥ 1). `--max-concurrent-runs` overrides it, and `FURROW_MAX_CONCURRENT_RUNS` is the fallback. |
| `FURROW_WEB_TOKEN` | unset | When set, `GET /` and every `start` message require the matching token. |

Example `.env`:

```dotenv
FURROW_PROVIDER=anthropic
FURROW_MODEL=claude-sonnet-4-20250514
FURROW_MAX_PARALLEL_TASKS=3
FURROW_MAX_CYCLES=5
FURROW_LOG_LEVEL=DEBUG
```

## Providers

| Provider | Selected with | Credentials |
| --- | --- | --- |
| `anthropic` (default) | `--provider anthropic` | `ANTHROPIC_API_KEY` or `FURROW_ANTHROPIC_API_KEY` |
| `openai` | `--provider openai` | `OPENAI_API_KEY` or `FURROW_OPENAI_API_KEY` |
| `ollama` | `--provider ollama` | none — point `FURROW_OLLAMA_BASE_URL` at your server |

```bash
export OPENAI_API_KEY=sk-...
furrow start "Add JWT auth to the API" --provider openai --model gpt-4o

# local model, no key
furrow start "Add JWT auth to the API" --provider ollama --model llama3
```

Select the provider with `--provider` or `FURROW_PROVIDER`. Each role can use a different
model through `--planner-model`, `--worker-model`, `--tester-model` or the matching
`FURROW_*_MODEL` settings; `--model` overrides all three at once.

**Retries are automatic for transient failures.** Each completion is attempted up to
`retry_attempts` times with exponential backoff based on `retry_min_wait`. Rate limits
(HTTP 429), overloaded responses, timeouts, connection errors and any 5xx are retried.
Authentication and invalid-request errors (400, 401, 403, 404 and similar) are terminal and
fail fast rather than burning the retry budget — check your key first if a run dies immediately.

Model output is parsed as JSON; the parser tolerates fenced blocks and JSON embedded in prose,
and reports a truncated snippet of the raw response when it cannot find an object.

## Logging

Logging is structlog-based: console renderer, ISO timestamps, context variables merged into
each record. `configure_logging(level=None)` in
[`furrow/config.py`](furrow/config.py) configures it, using `settings.log_level` unless a level
is passed. It is safe to call repeatedly — the last call wins and handlers are never duplicated.

Two ways to set the level:

```bash
furrow start "…" --log-level DEBUG      # CLI flag, on both commands
FURROW_LOG_LEVEL=DEBUG furrow start …  # environment or .env
```

Valid levels are `DEBUG`, `INFO` (default), `WARNING`, `ERROR` and `CRITICAL`, case-insensitive;
anything else is rejected as a configuration error. Each LLM call is logged at `INFO` with its
provider, model and prompt size, retries at `WARNING`, and `--log-level DEBUG` adds response
sizes and the debug records.

## Development

```bash
pip install -e ".[dev]"

pytest                    # test suite (asyncio_mode = auto, testpaths = tests)
ruff check                # lint
ruff format --check       # formatting
mypy                      # type check
```

The tools are configured in `pyproject.toml`: ruff uses a 100 character line length, targets
Python 3.10 and pins its rule set explicitly, and mypy runs in strict mode over `furrow/`, so
bare `mypy` needs no arguments.

**The test suite must not make real LLM calls, and the suite enforces it.** An autouse fixture
in [`tests/conftest.py`](tests/conftest.py) denies any non-loopback socket connection or DNS
lookup, so no test can reach a provider even by accident. Tests inject scripted replies into
`LLMClient.complete`, which means the real planner, worker and tester agents and the real
pydantic models are exercised end to end without a network. The same fixtures clear inherited
`FURROW_*` variables, fail if a test mutates the global `Settings` singleton, and undo the
process-wide logging changes.

## Project layout

```text
furrow/
├── __init__.py            # exports LLMClient and Settings
├── config.py              # Settings, Plan/TaskModel/TestResult, configure_logging
├── llm.py                 # provider transport, retries, JSON extraction, file IO
├── cli/
│   └── main.py            # click entry point: furrow start, furrow web
├── agents/
│   ├── planner.py         # goal -> Plan (1-5 dependency-aware tasks)
│   ├── worker.py          # task -> full file contents written to the workspace
│   ├── tester.py          # runs the test command, asks the model for a verdict
│   └── prompts.py         # system and task prompts, JSON contracts
├── core/
│   └── orchestrator.py    # the plan -> execute -> test cycle and its event stream
└── web/
    └── server.py          # FastAPI app: /, /health, /api/state and the /ws stream
```

See also: [`furrow/config.py`](furrow/config.py) for every setting and its default,
[`furrow/cli/main.py`](furrow/cli/main.py) for the exact command surface, and
[`furrow/web/server.py`](furrow/web/server.py) for the websocket protocol.