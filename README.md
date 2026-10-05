# Furrow

## What it is

Furrow is an autonomous coding agent. You give it a goal; it turns that goal into a
task plan, runs the tasks in parallel, runs your project's test suite, and then repeats
the whole cycle until the goal is met or a stop condition fires.

Furrow ships two front ends over the same loop: `furrow start` works in the terminal and
`furrow web` serves a browser UI that streams the loop's events over a websocket.

Scope note, so nothing here is oversold: the current worker agent prompts a model and
returns its report as text — there is no file-editing tool loop yet. The test side is
real, though: the tester agent genuinely executes your project's test runner.

## Install

Python 3.10 or newer is required.

```bash
pip install furrow
```

For a development checkout, install in editable mode with the `dev` extra
(`pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`):

```bash
pip install -e ".[dev]"
```

A `lint` extra (`ruff` + `mypy` only) exists for CI jobs that do not run the tests.

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow web
```

`start` takes the goal as an optional argument and prompts for it when omitted.

### `furrow start`

| Option | Effect |
| --- | --- |
| `--model` | Model for every agent. Overrides `FURROW_MODEL`, `FURROW_PLANNER_MODEL`, `FURROW_WORKER_MODEL` and `FURROW_TESTER_MODEL` together. |
| `--planner-model` | Model for the planner agent only. Applied after `--model`, so it wins. |
| `--worker-model` | Model for worker agents only. Applied after `--model`, so it wins. |
| `--tester-model` | Model for the tester agent only. Applied after `--model`, so it wins. |

### `furrow web`

| Option | Effect |
| --- | --- |
| `--host` | Interface to bind. Defaults to `FURROW_HOST` (`127.0.0.1`). |
| `--port` | Port to listen on. Defaults to `FURROW_PORT` (`8000`). |

`furrow --version` prints the installed distribution version, and `-h` / `--help` works
on the group and on each subcommand.

`start` exits with `0` on success, `2` for a configuration problem such as a missing API
key, `1` for an unexpected runtime error (reported as a single `Error: ...` line rather
than a traceback), and `130` when interrupted with Ctrl-C.

## Configuration

Every setting is read from the environment with the `FURROW_` prefix, and a `.env` file
in the working directory is read as well (`env_file=".env"`). Real environment variables
take precedence over `.env`.

| Variable | Default | Description |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | Provider to call: `anthropic`, `openai` or `ollama`. |
| `FURROW_MODEL` | `claude-sonnet-5` | Fallback model used when a caller does not request one. |
| `FURROW_PLANNER_MODEL` | `claude-haiku-4-5-20251001` | Model used to turn the goal into a task plan. |
| `FURROW_WORKER_MODEL` | `claude-sonnet-5` | Model used to implement a single task. |
| `FURROW_TESTER_MODEL` | `claude-sonnet-5` | Model used to review the workspace and judge the tests. |
| `FURROW_ANTHROPIC_API_KEY` | unset | Anthropic API key. See [API keys](#api-keys). |
| `FURROW_OPENAI_API_KEY` | unset | OpenAI API key. See [API keys](#api-keys). |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL of the Ollama server, with no `/api` suffix; the client appends the path. |
| `FURROW_OLLAMA_MODEL` | `llama3.1` | Ollama model tag. Used as the completion model when the provider is `ollama`. |
| `FURROW_WORKSPACE` | the current working directory | Root directory the agents read from and write to. |
| `FURROW_MAX_CYCLES` | `0` | Maximum plan/execute cycles. `0` means unlimited. |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Maximum number of worker tasks executed concurrently. |
| `FURROW_MAX_CONSECUTIVE_FAILURES` | `3` | Circuit breaker: stop after this many failing cycles in a row. `0` or less disables it. |
| `FURROW_MAX_TOKENS` | `4096` | Maximum tokens requested per LLM completion. |
| `FURROW_LLM_TIMEOUT_SECONDS` | `120.0` | Per-request timeout, in seconds, for every LLM call. |
| `FURROW_LLM_MAX_RETRIES` | `3` | Attempts per LLM call, including the first one. `1` disables retries. |
| `FURROW_LOG_LEVEL` | `INFO` | Log level applied by the CLI and web entry points. |
| `FURROW_HOST` | `127.0.0.1` | Interface the web server binds to. |
| `FURROW_PORT` | `8000` | Port the web server listens on. |

> **Model IDs are defaults, not guarantees.** The IDs in the table are the values this
> release ships with. They are overridden by `FURROW_MODEL`, the `FURROW_*_MODEL`
> variables, or the `--model` / `--*-model` flags, so confirm the IDs your own account
> can actually reach before relying on them — model names and availability change, and
> an unreachable ID surfaces as a provider error on the first call.

## Providers

Set with `FURROW_PROVIDER`.

- `anthropic` (default) — talks to the Anthropic Messages API through the `anthropic` SDK.
- `openai` — talks to the OpenAI chat completions API through the `openai` SDK.
- `ollama` — talks to a local Ollama server over plain HTTP, using
  `FURROW_OLLAMA_BASE_URL` (default `http://localhost:11434`) and `FURROW_OLLAMA_MODEL`
  (default `llama3.1`).

Every LLM call is retried on transient failures — connection and timeout errors, HTTP
408/429/5xx, and the SDKs' rate-limit and connection errors — up to
`FURROW_LLM_MAX_RETRIES` attempts with exponential backoff, and each attempt is bounded
by `FURROW_LLM_TIMEOUT_SECONDS`.

## API keys

Both forms work, and the prefixed form is checked first:

- Anthropic: `FURROW_ANTHROPIC_API_KEY`, falling back to the unprefixed
  `ANTHROPIC_API_KEY`.
- OpenAI: `FURROW_OPENAI_API_KEY`, falling back to the unprefixed `OPENAI_API_KEY`.

`furrow start` checks for the configured provider's key before starting and exits `2`
with an actionable message if neither variable is set. Ollama needs no key.

A `.env` file in the working directory is read, which is the convenient way to keep a key
out of your shell history:

```bash
# .env  (never commit this)
ANTHROPIC_API_KEY=sk-ant-...
```

Keep `.env` out of git and commit a `.env.example` template instead — the repository's
`.gitignore` already does that for you.

## How the loop works

One cycle is:

1. **Plan** — the planner agent turns the goal into 1–5 tasks, each with an id, a
   description, the files to touch and its dependencies. The reply must validate against
   the `Plan` schema; a reply that cannot be parsed fails the cycle.
2. **Execute** — one worker agent per task, run concurrently, at most
   `FURROW_MAX_PARALLEL_TASKS` at a time. Tasks are awaited with exceptions captured
   rather than raised, so one failing task does not cancel its siblings; it is recorded
   as a failed task.
3. **Test** — the tester agent runs the first usable test runner in `FURROW_WORKSPACE`
   (`pytest`, `python -m pytest` when pytest is importable, `npm test`, `pnpm test`,
   `yarn test`, `cargo test`, `go test`), capped at 120 seconds per runner. Only exit
   codes `0` and `1` count as a verdict; anything else means that runner cannot speak
   for the project and the next candidate is tried. The runner's output, its exit code
   and the per-task worker reports go to the tester model, which returns a `TestResult`.
   An unparseable tester reply fails closed as `passed: false` — the verdict is never
   inferred from free text.
4. **Cycle end** — if the tests failed, their failure messages are appended to the goal
   so the next cycle has something concrete to act on, and the loop waits with an
   exponential backoff (1 second, doubling, capped at 30 seconds) before retrying.

The loop stops when any of these happens:

- **Goal complete** — the tests passed and every planned task completed. This is the
  normal exit; the halt reason is printed and, in the web UI, streamed as a `halt` event.
- **`FURROW_MAX_CYCLES`** — reached, if it is greater than `0`. The default `0` means
  unlimited cycles.
- **`FURROW_MAX_CONSECUTIVE_FAILURES`** — the circuit breaker. A cycle counts as failed
  when any task raised or the test verdict failed. When that many failures accumulate in
  a row the loop halts instead of spinning. `0` or less disables the breaker.
- **You stop it** — Ctrl-C in the terminal (`130`), or the Stop button / socket close in
  the web UI.

If the planner returns no tasks at all, the loop only finishes if the previous cycle's
tests passed; otherwise the empty plan counts as a failure.

## Web UI

```bash
furrow web
```

Serves on `127.0.0.1:8000` by default (`FURROW_HOST` / `FURROW_PORT`, or
`furrow web --host ... --port ...`).

- `GET /health` — liveness probe returning JSON: `status`, `provider`, `host`, `port`.
- `GET /` — the monitoring page. Its HTML, CSS and JavaScript are inline, so there is no
  static-file directory to configure.
- `/ws` — a websocket that runs one loop per connection. The client sends
  `{"goal": "...", "model": "..."}` as the first frame (`model` is optional and points
  every agent at that model). The server then streams JSON events as they happen:
  `cycle_start`, `plan`, `task_start`, `task_completed`, `task_failed`, `test_result`,
  `cycle_end`, `halt` and `error`. Sending `{"stop": true}` or closing the socket stops
  the loop, and the server tears the run down so a closed browser tab cannot leave an
  agent loop running in the background. An invalid first frame or a failed run is
  reported to the client as an `error` frame with a matching websocket close code.

The default loopback bind is deliberate: `/ws` starts agents that run project commands on
this host, so only widen it when you have decided that is acceptable.

## Development

```bash
pytest            # the test suite (asyncio_mode = auto, testpaths = tests)
pytest --cov      # with coverage; the configured floor is 80%, so a lower run fails
ruff check .      # lint
ruff format .     # format
mypy furrow       # type check
```