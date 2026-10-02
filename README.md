# Furrow

Autonomous coding agent with an infinite parallel development loop. Give it a goal and
it plans file-level changes, executes the planned tasks with parallel workers, runs the
project's test suite, feeds the failures back into the next cycle, and repeats until the
goal is reached.

## Install

```bash
pip install furrow         # published package, Python 3.10+
pip install -e ".[dev]"    # from a checkout, with pytest, ruff and mypy
```

## How it works

A run is a sequence of cycles, each one plan → execute → test.

1. **Plan.** The planner turns the goal into at most `FURROW_PLANNER_MAX_TASKS` (5)
   tasks, each with an id, a description, the files to change and dependencies. It also
   sees the previous cycle's failures, the tasks already completed, the last test verdict
   and a truncated listing of the repository. An empty task list means the goal is
   already satisfied. The goal itself is never rewritten.
2. **Execute.** Tasks are grouped into dependency waves (a topological order); tasks
   within a wave run concurrently, bounded by `FURROW_MAX_PARALLEL_TASKS` (5, and 0 or
   less falls back to one at a time). One worker executes one task, taking exactly one
   action per turn — `read`, `list`, `write` or `finish` — for at most
   `FURROW_WORKER_MAX_TURNS` (12) turns. Newly planned tasks are merged by description,
   so work completed in an earlier cycle is skipped and work that failed is retried. A
   task is abandoned as failed after `FURROW_MAX_TASK_ATTEMPTS` (3) failed attempts.
3. **Test.** The tester detects the project's test command (pytest, npm/pnpm/yarn, cargo
   or go), runs it from the workspace root, and takes the verdict from the real exit
   code. A failing run is summarized, and its failures become the focus for the next
   planning round.

The loop stops for exactly these reasons (`stop_reason`):

| `stop_reason` | Meaning |
| --- | --- |
| `goal complete` | every task in the last plan is completed or skipped, and the last test run passed — or found no test command, which counts as unverifiable rather than failing |
| `planner reported no remaining work` | the planner returned an empty plan and the last test run passed |
| `cycle limit reached` | `FURROW_MAX_CYCLES` cycles have run; `0` means unlimited |
| `no progress` | two consecutive cycles each completed no task — the test verdict is irrelevant, so a green run over tasks that all failed still counts as no progress, which is what keeps `FURROW_MAX_CYCLES=0` from looping forever; the terminal reasons above are checked first, so a cycle that converges by skipping earlier work is still reported as complete |
| `repeated cycle errors: ...` | three consecutive cycles raised (planner error, provider error, …) |
| `stop requested` | `Orchestrator.stop()` was called, for example by Ctrl-C |

Progress is published as events — `cycle`, `plan`, `task_start`, `task_done`,
`task_failed`, `task_skipped`, `tests`, `log`, `error`, `done` — which the CLI prints and
the web server streams. `done` is always last and carries the run summary.

## Configuration

Every setting is read from a `FURROW_`-prefixed environment variable and from a `.env`
file in the working directory; command-line flags win over both.

| Variable | Default | Meaning |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | `anthropic`, `openai` or `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | fallback model when a role has none |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | model used by the planner |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | model used by the workers |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | model used to summarize test failures |
| `FURROW_ANTHROPIC_API_KEY` | unset | falls back to `ANTHROPIC_API_KEY` in the environment |
| `FURROW_OPENAI_API_KEY` | unset | falls back to `OPENAI_API_KEY` in the environment |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | server URL for `FURROW_PROVIDER=ollama` |
| `FURROW_OLLAMA_MODEL` | `qwen2.5-coder:7b` | fallback model for the `ollama` provider |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | workers running concurrently; 0 or less means one |
| `FURROW_MAX_CYCLES` | `0` | cycle limit; 0 means unlimited |
| `FURROW_WORKSPACE` | current directory | sandbox root that bounds every worker file operation |
| `FURROW_PLANNER_MAX_TASKS`, `FURROW_WORKER_MAX_TURNS` | `5`, `12` | cap on the tasks in one plan / action turns per worker |
| `FURROW_MAX_TASK_ATTEMPTS` | `3` | dispatches per task; a task that has used them up is marked failed as `abandoned after N attempts` and is never retried |
| `FURROW_TEST_COMMAND_TIMEOUT` | `180.0` | seconds before a test run is killed (reported as exit 124) |
| `FURROW_TEMPERATURE`, `FURROW_MAX_TOKENS` | `0.2`, `4096` | sampling temperature (not sent to Anthropic) / response cap |
| `FURROW_MAX_RETRIES` | `3` | attempts per completion; rate limits, timeouts and 5xx are retried with exponential backoff, auth and bad-request errors are not |
| `FURROW_REQUEST_TIMEOUT` | `120.0` | provider request timeout in seconds |
| `FURROW_MAX_OUTPUT_CHARS` | `20000` | cap on captured test output and stored task results |
| `FURROW_LOG_LEVEL` | `INFO` | declared in `Settings`; no code reads it yet |

With `FURROW_PROVIDER=ollama` the client talks to a local server through its
OpenAI-compatible API. The role models default to Anthropic model ids, so set
`FURROW_PLANNER_MODEL` / `FURROW_WORKER_MODEL` / `FURROW_TESTER_MODEL` to a local name too.

## Usage

```bash
furrow start "Add JWT auth to the API" \
  --provider anthropic --planner-model claude-3-5-haiku-20241022 \
  --max-parallel 4 --max-cycles 5 \
  --workspace ~/code/api --verbose
```

`furrow start [GOAL]` prompts for the goal when it is omitted.

| `start` option | Meaning |
| --- | --- |
| `--model` | override the model for every role at once |
| `--planner-model`, `--worker-model`, `--tester-model` | override one role; wins over `--model` |
| `--provider` | `anthropic`, `openai` or `ollama` |
| `--max-cycles` | cycle limit, `0` (the default) means unlimited |
| `--max-parallel` | maximum tasks running concurrently |
| `--workspace` | workspace root; defaults to `FURROW_WORKSPACE` |
| `--verbose` / `--quiet` | also print `log` events / print failures only |

Exit codes: `0` the goal was completed, `1` the run finished without completing it or
failed outright, `130` stopped with Ctrl-C (the run then stops cooperatively at the next
checkpoint). The web UI and the version:

```bash
furrow web --host 127.0.0.1 --port 8000
furrow --version
```

## Web UI

| Route | Purpose |
| --- | --- |
| `GET /` | the self-contained UI: a goal field, an optional model field, a live log |
| `GET /api/health` | `{"status": "ok", "version": "..."}` |
| `GET /api/config` | effective provider and model names, never any credentials |
| `WS /ws` | send `{"goal": "...", "model": "..."}` once, then read the run |

The browser sends the payload when the socket opens; a `model`, when given, applies to
every role. The server then streams one JSON event per line — `kind`, `message`, `cycle`,
`task_id`, `data`, `ts` — until the run ends, the browser goes away (which stops the run
cooperatively) or the server shuts down.

The server binds `127.0.0.1:8000` by default, on purpose: an endpoint that can spend
your API key and rewrite your workspace should not be reachable from the network by
accident. Pass `--host 0.0.0.0` only if you have decided that is acceptable.

## Architecture

| File | Responsibility |
| --- | --- |
| `furrow/config.py` | pydantic models (`Settings`, `Plan`, `TaskModel`, `TestResult`, `RunResult`) and the `FURROW_`-prefixed settings |
| `furrow/llm.py` | provider clients (Anthropic, OpenAI, Ollama), retries and backoff, sandboxed file read/write and workspace listing |
| `furrow/util.py` | tolerant JSON extraction from model replies, truncation, and `resolve_within` — the path sandbox |
| `furrow/core/orchestrator.py` | the cycle loop: plan, dependency waves, bounded parallel execution, test feedback, stop conditions |
| `furrow/core/events.py` | the event contract shared by the orchestrator, the CLI, the web server and the tests |
| `furrow/agents/planner.py` | goal plus focus into a small, normalized plan; retries once, then raises `PlannerError` |
| `furrow/agents/worker.py` | one task through a bounded `read`/`list`/`write`/`finish` loop, sandboxed to the workspace |
| `furrow/agents/tester.py` | test-runner detection, running it with a timeout, and turning its exit code into a verdict |
| `furrow/agents/prompts.py` | the prompt and JSON schema text for the three agents |
| `furrow/cli/main.py` | click commands, settings overrides, event printing, exit codes |
| `furrow/web/server.py` | FastAPI app, the static page, and the WebSocket event stream |

## Development

```bash
pytest
ruff check .
mypy furrow
```

`pytest` runs in `asyncio_mode = auto` over `tests/`; pytest, ruff and mypy are all
configured in `pyproject.toml`. The suite is fully offline: agents are driven with stub
agents and a fake LLM client, provider SDK clients are replaced with mocks, and the only
subprocesses spawned are `sys.executable`.

## Safety

- Workers may only touch the workspace. Every read, list and write is resolved against
  `FURROW_WORKSPACE` (default: the current directory), and a path that escapes it — an
  absolute path, or one using `..` — is refused and fed back to the model as an error
  instead of being written.
- Test commands run as a subprocess, without a shell, from the workspace root, and are
  killed after `FURROW_TEST_COMMAND_TIMEOUT`. Whatever the project's own test suite does,
  it does with your user's permissions.
- Running furrow means giving an LLM write access to that directory. Writes overwrite
  whole files, there is no rollback, and nothing is guarded beyond the workspace boundary:
  use a branch or a copy.
- Runs are not network-sandboxed. Workers have no shell and no network themselves, but
  the agents call model providers over the network, and a project with no detectable test
  runner is reported as "verification skipped" rather than verified.
