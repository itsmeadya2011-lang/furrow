# Furrow

Autonomous coding agent that runs a plan / execute / verify loop against a real
workspace. A planner agent turns a goal into tasks, worker agents edit the
repository through a sandboxed tool layer, and a tester agent runs the project's
own tests, linter and type checker. Failures from one cycle become the focus of
the next one.

## Install

Python 3.10 or newer.

```bash
pip install -e .            # runtime
pip install -e ".[dev]"     # runtime + pytest, ruff, mypy
```

## Quick start

```bash
export FURROW_ANTHROPIC_API_KEY=sk-...

furrow plan "Add JWT auth to the API"      # planner only, one LLM call
furrow start "Add JWT auth to the API"     # full loop
furrow web                                 # monitoring UI + REST + websocket
furrow --version
```

`furrow start` prompts for the goal when none is given. Ctrl+C once asks the
orchestrator to stop gracefully at the next cycle boundary and still prints the
run summary; a second Ctrl+C cancels the run.

### `furrow start [GOAL]`

| Option | Effect |
| --- | --- |
| `--model` | Model for every agent |
| `--planner-model`, `--worker-model`, `--tester-model` | Per-role overrides |
| `--provider` | `anthropic`, `openai` or `ollama` |
| `--workspace` | Directory the agent may touch (default: cwd) |
| `--max-cycles` | Cycle cap; `0` means unlimited |
| `--max-parallel` | Concurrent tasks per wave (default 5) |
| `--max-tool-iterations` | Model/tool rounds per task (default 12) |
| `--state-file` | Write a JSON state snapshot after every cycle |
| `--log-level` | Log verbosity |
| `--no-live-output` | Suppress the live event feed |

`--model` sets all four model settings; a per-role flag still wins. Events are
rendered as the run proceeds, and the run summary (cycles, tasks, last test
verdict, halt reason, token usage) is printed at the end.

### `furrow plan GOAL`

Runs the planner only and prints the task table plus its rationale. No workers,
no tests, one planning request.

### `furrow web`

```bash
furrow web --host 127.0.0.1 --port 8000 --reload
```

Defaults come from `FURROW_WEB_HOST` (`127.0.0.1`) and `FURROW_WEB_PORT`
(`8000`). `--reload` enables uvicorn auto-reload.

## Configuration

Every setting in `furrow/config.py` is read from the environment with the
`FURROW_` prefix, and a `.env` file in the working directory is loaded
automatically (see `.env.example`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | `anthropic`, `openai`, `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Fallback model |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Planner model |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Worker model |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Tester model |
| `FURROW_ANTHROPIC_API_KEY` | unset | Falls back to `ANTHROPIC_API_KEY` |
| `FURROW_OPENAI_API_KEY` | unset | Falls back to `OPENAI_API_KEY` |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Concurrency cap |
| `FURROW_MAX_CYCLES` | `0` | Cycle cap, `0` = unlimited |
| `FURROW_MAX_STALLED_CYCLES` | `3` | Halt after N no-progress cycles |
| `FURROW_MAX_TOOL_ITERATIONS` | `12` | Tool rounds per worker |
| `FURROW_WORKSPACE` | cwd | Sandbox root |
| `FURROW_TEST_COMMAND` | auto-detected | Overrides the test command |
| `FURROW_LINT_COMMAND` | auto-detected | Overrides the lint command |
| `FURROW_TYPECHECK_COMMAND` | auto-detected | Overrides the typecheck command |
| `FURROW_TEST_TIMEOUT` | `300` | Seconds per verification command |
| `FURROW_STATE_FILE` | unset | State snapshot path |
| `FURROW_WEB_HOST` / `FURROW_WEB_PORT` | `127.0.0.1` / `8000` | Web bind |
| `FURROW_TEMPERATURE` / `FURROW_MAX_TOKENS` | `0.0` / `8192` | Sampling |
| `FURROW_MAX_RETRIES` | `3` | Retries for transient LLM failures |
| `FURROW_REQUEST_TIMEOUT` | `120.0` | Seconds per LLM request |

The bare `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` still work as a fallback for
the corresponding `FURROW_` setting. Ollama needs no key; its base URL is used
as an OpenAI-compatible endpoint (`/v1` is appended), and a configured model
that names an Anthropic model falls back to `llama3.1`.

Transient LLM failures (429, connection errors, timeouts, 5xx) are retried with
exponential backoff and jitter; authentication and other 4xx errors are not
retried. An exhausted rate limit raises `LLMRateLimitError`.

## How the loop terminates

`Orchestrator.run()` returns a `RunSummary` and stops at the first halt
condition it meets:

| `halted_reason` | Cause |
| --- | --- |
| `stopped` | `stop()` was called (Ctrl+C, REST stop) |
| `max_cycles` | `max_cycles > 0` and the cap was reached |
| `stalled` | `max_stalled_cycles` consecutive cycles made no progress |
| `goal_complete` | The planner returned an empty task list |

The original goal is never overwritten: failures accumulate into a separate,
bounded focus string that is passed to the next planner call. Planner and tester
exceptions are isolated into `ERROR` events and a stalled cycle instead of
killing the run.

With `FURROW_STATE_FILE` set, a JSON state snapshot is written atomically after
every cycle. **Resuming a run from a state file is not implemented** - a new
process starts a fresh run.

## Sandbox guarantee

Every worker capability lives in `furrow/agents/tools.py` and goes through one
registry: `read_file`, `write_file` (atomic temp file plus rename), `edit_file`
(exact-string replace; refuses a missing or ambiguous match unless
`replace_all`), `list_files`, `search` and `run_command`.

Every path is resolved against the workspace root and refused when it escapes -
`../` traversal, an absolute path outside the root, or a symlink pointing out.
Tool errors are returned to the model as readable text rather than raised. Each
worker is bounded by `max_tool_iterations`; hitting the cap returns an explicit
`INCOMPLETE` summary instead of reporting a false success.

### What a `tool_call` event may contain

A `tool_call` event reaches four places: the `EventBus` ring buffer,
`GET /api/runs/{id}`, `GET /api/runs/{id}/events` and the `/ws` live stream. Its
`message` is built by an explicit allowlist, `PREVIEW_SAFE_ARGS` in
`furrow/agents/tools.py`, and `PREVIEW_CHARS` (240) caps the **whole** assembled
message, tool name included, so `len(message) <= 240` always.

What the allowlist guarantees:

| Argument | Reaches the event feed? | Why |
| --- | --- | --- |
| `write_file.content` | **never** | File content by definition |
| `edit_file.old_string` / `new_string` | **never** | File content by definition |
| `search.query` | **never** | Literal substring search: the query is file text |
| `run_command.command` | yes - message (clipped) and `data["command"]` | Deliberate; see below |
| `path`, `pattern`, `glob`, `file`, `dir` | yes | Model-named identifiers, not content |
| `replace_all`, `max_bytes`, `max_results`, `timeout` | yes | Booleans and numbers |

So the precise guarantee is: **the content arguments of `write_file` and
`edit_file`, and the literal search text of `search`, never appear in a
`tool_call` message, in its `data` payload, or in `Event.to_dict()`** - on the
success path and on every error path.

The guarantee is scoped to `tool_call` arguments, not to the whole feed. Other
event types carry their own prose: a `task_end` snapshot holds the worker's
summary, which is the model's own words, and `test_result` holds the tester's
verdict. Neither is derived from a tool argument, but a model that pastes file
text into its own summary can still put it in the run snapshot. Treat the event
feed as sensitive if the workspace is.

`run_command`'s command text is the one deliberate exception, and it is not an
oversight: it is the most useful live observability signal there is, it is
clipped to 60 characters head/tail, and it is echoed in `event.data["command"]`
by the tool's own detail payload, so removing it from the message would not close
the route - it would only make it harder to notice. The operator-visible
consequence: **never embed a secret in a shell command** (`sed`, `echo`, a
heredoc), because the command line is in the event feed. The same applies to file
*names*: a `path` renders verbatim, so a credential stashed in a filename is in
the feed too.

Identifier arguments are rendered intentionally - they are what makes the feed
readable at all (`search(path=.)` next to `{"matches": 3}` tells an operator the
search was aimed at the workspace root and hit three lines, which is the fact
that matters; the query text itself is withheld). Only keys the caller actually
sent are rendered, so a `search` that omits `path` reads `search()` rather than
inventing the tool's `"."` default. The allowlist is enforced from the argument
NAME, not from inspecting the value, so a dict or list wrapped around an
allowlisted name is skipped rather than stringified.

## Verification

After every cycle the tester runs the test command, then the linter and the type
checker. Each one is either taken verbatim from `FURROW_TEST_COMMAND` /
`FURROW_LINT_COMMAND` / `FURROW_TYPECHECK_COMMAND`, or auto-detected from an
executable on PATH plus a project marker file such as `pyproject.toml` or
`package.json`. Commands run inside the resolved workspace, with no shell, under
`FURROW_TEST_TIMEOUT`.

The verdict is the ground truth, not the model: if the model claims a pass while
a command exited non-zero or timed out, the result is overridden to failed and
the offending command is named in `failures`. A timeout is a failure, never a
fall-through to the next candidate. `TestResult.commands` lists what actually
ran. If no test command can be resolved, verification is reported as skipped
with an empty command list rather than as a pass.

## Web UI

`furrow web` serves a single-page monitoring UI from the same process. The page
is served without a token; `/api/*` and the websocket require one whenever
`FURROW_WEB_TOKEN` is set.

| Route | Behaviour |
| --- | --- |
| `GET /` | Single-page monitoring UI |
| `GET /api/health` | Status, version, run counters |
| `GET /api/runs` | Active and recent runs |
| `POST /api/runs` | Start a run in the background (202); 400 on a bad goal, 429 over 8 concurrent |
| `GET /api/runs/{id}` | State snapshot plus recent events |
| `POST /api/runs/{id}/stop` | Graceful stop request |
| `GET /api/runs/{id}/events` | Buffered events for polling |
| `WS /ws` | Live event stream |
| `GET /docs`, `GET /redoc`, `GET /openapi.json` | Swagger UI, ReDoc and the schema; served only while `FURROW_WEB_TOKEN` is unset |

The websocket takes one JSON message that picks the mode: `{"goal": "..."}`
starts a run and streams its events, `{"run_id": "..."}` attaches to an existing
run, replays its buffered events and starts nothing. Control frames carry
`kind: "control"` (`started`, `done`, `error`); branch on `kind`, not on `type`,
because one event type is literally named `error`.

### Security

The server starts runs that execute commands inside the workspace and spend real
LLM credits, so set `FURROW_WEB_TOKEN` for any non-loopback deployment:

```bash
export FURROW_WEB_TOKEN=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))")
furrow web --host 0.0.0.0
```

`/api/*` then requires `Authorization: Bearer <token>` and the websocket
handshake requires `?token=<token>`, because browsers cannot set headers on a
websocket handshake. A prominent warning is logged at startup while the server is
unauthenticated.

A **blank** token counts as no token. `FURROW_WEB_TOKEN=""` and
`FURROW_WEB_TOKEN="   "` are both treated as unset: leading and trailing
whitespace is stripped, and an empty result means the gate is disabled. A stray
trailing space in a systemd unit or a `.env` line would therefore leave the whole
API open on a deployment the operator believed was protected, so this case gets
its own `ERROR`-level startup warning naming the variable, in addition to the
generic "no authentication" warning. Unset the variable if you want an
unauthenticated server on purpose; never leave it set to whitespace.

The interactive docs are the one deliberate exception, and they are closed rather
than gated. `/docs`, `/redoc` and `/openapi.json` are served normally while no
token is configured - the zero-config loopback experience is unchanged - but as
soon as `FURROW_WEB_TOKEN` is set they answer `401` without a credential and
`404` even with a valid one: they are not served at all. The reason is that
Swagger UI fetches `openapi.json` from the same origin in a browser request that
carries no `Authorization` header, so a header-gated schema would leave `/docs`
rendering as a broken "failed to load API definition" page, and publishing the
schema to token holders alone still discloses a map of a surface that executes
shell commands. Browse the docs on an unauthenticated loopback run, and script
against `/api/*` with `curl -H "Authorization: Bearer $FURROW_WEB_TOKEN"` on a
tokened deployment.

## Development

```bash
python3 -m pytest -q        # offline suite: no network, no virtualenv
python3 -m ruff check .
python3 -m mypy furrow
```

`tests/conftest.py` provides `ScriptedLLMClient`, a no-network LLM double, plus
the `workspace`, `settings`, `bus`, `fake_client` and `make_client` fixtures.
Verification commands run inside a `tmp_path` workspace, so the suite never
touches the repository.

## Architecture

```
furrow/
  cli/main.py            # furrow start | web | plan
  config.py              # Settings (FURROW_ env prefix), TaskModel, Plan, TestResult
  core/orchestrator.py   # the cycle loop, halt conditions, RunSummary
  agents/planner.py      # goal -> validated Plan (untrusted JSON is normalised)
  agents/worker.py       # one task, driven through the tool registry
  agents/tester.py       # test + lint + typecheck, verdict gated on exit codes
  agents/tools.py        # sandboxed tool layer
  agents/prompts.py      # planner, worker and tester system prompts
  llm.py                 # anthropic / openai / ollama, JSON repair, retries, usage
  events.py              # EventBus: event types, ring buffer, subscriber queues
  web/server.py          # REST API, websocket stream, bundled UI
```

Agents run tasks in dependency waves: tasks whose `dependencies` are satisfied
start together under an `asyncio.Semaphore` bounded by `max_parallel_tasks`, and
a later wave waits for the ones it depends on.