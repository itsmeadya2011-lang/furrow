# Furrow

An autonomous coding agent that runs a development loop: it plans a goal into
tasks, executes the tasks in parallel, runs the project's tests, and repeats
until the goal is satisfied or you stop it.

## Requirements

- Python 3.10+ (`requires-python = ">=3.10"`; the code uses `X | Y` type syntax)
- An API key for the provider you configure (see [API keys](#api-keys))

## Install

There is no PyPI release for this project. The `furrow` name on PyPI belongs to
an unrelated text-segmentation library, so `pip install furrow` installs that
package instead. Install from a checkout:

```bash
pip install .
```

This installs the `furrow` console script. For a development checkout:

```bash
pip install -e ".[dev]"
```

## Usage

### `furrow start "<goal>"`

Runs the loop in the terminal, writing only inside the workspace (the working
directory by default). Omitting the goal prompts for one.

```bash
furrow start "Add JWT auth to the API"
furrow start                      # prompts for the goal if omitted
furrow start "<goal>" --model claude-opus-4-20250514
```

| Option | Default | Description |
| --- | --- | --- |
| `--provider TEXT` | `FURROW_PROVIDER` (`anthropic`) | `anthropic`, `openai`, or `ollama` |
| `--model MODEL` | `FURROW_MODEL` | Model for every role; an explicit `--<role>-model` wins |
| `--planner-model MODEL` | `FURROW_PLANNER_MODEL` | Model for planning |
| `--worker-model MODEL` | `FURROW_WORKER_MODEL` | Model for writing files |
| `--tester-model MODEL` | `FURROW_TESTER_MODEL` | Model for summarizing test output |
| `--max-cycles N` | `FURROW_MAX_CYCLES` (`0`) | Stop after N cycles; `0` means unlimited |
| `--parallel N` | `FURROW_MAX_PARALLEL_TASKS` (`5`) | Tasks in flight at once (`max_parallel_tasks`) |
| `--workspace DIR` | `FURROW_WORKSPACE` (current directory) | The only tree the agents may read and write |
| `--log-level LEVEL` | `FURROW_LOG_LEVEL` (`INFO`) | stdlib log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--debug` | off | Print a traceback when a run fails unexpectedly |

Exit codes: `0` only when the loop stopped because the goal was complete (tests
passed and no task failed); `1` when it stopped for any other reason or the run
errored; `2` for a usage error such as an empty goal; `130` on `Ctrl-C`, which
stops the loop and its in-flight work.

### `furrow web`

Serves the same loop behind a small browser UI.

```bash
furrow web              # http://127.0.0.1:8000
furrow web --port 9000
```

| Option | Default | Description |
| --- | --- | --- |
| `--host HOST` | `127.0.0.1` | Interface to bind |
| `--port PORT` | `8000` | TCP port to listen on |
| `--log-level LEVEL` | `info` | uvicorn log level: `critical`, `error`, `warning`, `info`, `debug`, `trace` |

**There is no authentication.** This endpoint drives an agent that runs
subprocesses and writes files, so the default binds loopback only. Any other
`--host` publishes that endpoint to the network and logs a warning at startup;
do not expose it to a network you do not trust.

What it serves:

- `GET /` — the single-page UI: a goal field and a live log.
- `GET /ws` — the run's websocket. The client sends one start message,
  `{"goal": "...", "model": "..."}`, where `model` is optional and applies to
  all three roles. The server then sends one JSON message per orchestrator
  event — `run.start`, `cycle.start`, `plan`, `task.start`, `task.done`,
  `task.failed`, `task.skipped`, `test.result`, `cycle.done`, `run.done` —
  each shaped `{"type": ..., "cycle": <int|null>, "message": ...}` plus that
  event's extra keys (`passed`/`failures` for `test.result`, `reason`/`cycles`
  for `run.done`). A start message the server rejects gets a terminal `error`
  message and no run is started.
- `POST /api/start` — validates a goal and echoes the settings a run would use.
  It never starts a run; runs only stream over `/ws`.

The page renders the stream live: a line for each cycle start, for each plan (with
the number of tasks planned), and for each cycle end; one line per task as it
starts, completes (with the task's file list), fails, or is skipped; test results
colored by the exit-code verdict with the failure lines indented under them; and
a closing line with the cycle count and the stop reason. Every line is written
with `textContent`, so markup in model-derived text is displayed rather than
executed.

There is no stop button: closing the tab (or any websocket disconnect) cancels
the run, and the orchestrator cancels its in-flight workers on the way out, so
an abandoned run stops spending API credits.

## Configuration

Every setting is read from the environment with the `FURROW_` prefix, or from a
`.env` file in the working directory. The `furrow start` flags above override the
matching variables for that run; the rest can only be set through the
environment or `.env`.

| Variable | Default | Description |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | `anthropic`, `openai`, or `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Model for general calls |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used to write plans |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used to edit files |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used to read test output |
| `FURROW_ANTHROPIC_API_KEY` | unset | Anthropic key (see below) |
| `FURROW_OPENAI_API_KEY` | unset | OpenAI key (see below) |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Workers run concurrently, capped by this |
| `FURROW_MAX_CYCLES` | `0` | Cycle limit; `0` means unlimited |
| `FURROW_WORKSPACE` | current directory | The only tree the agents may write to |
| `FURROW_LOG_LEVEL` | `INFO` | Logging level for library logs |
| `FURROW_MAX_TOKENS` | `8192` | Max tokens per LLM response |
| `FURROW_REQUEST_TIMEOUT` | `120` | Per-request HTTP timeout, seconds |
| `FURROW_MAX_RETRIES` | `3` | Attempts per LLM call, for transient errors only (connection, timeout, 429, 5xx); `1` disables retrying |
| `FURROW_TEST_TIMEOUT` | `300` | Hard timeout for a test run, seconds |

Example `.env`:

```
FURROW_PROVIDER=anthropic
FURROW_MODEL=claude-sonnet-4-20250514
FURROW_MAX_PARALLEL_TASKS=3
```

## API keys

Set a key for whichever provider you use:

```bash
export ANTHROPIC_API_KEY=...   # or FURROW_ANTHROPIC_API_KEY
export OPENAI_API_KEY=...      # or FURROW_OPENAI_API_KEY, for provider=openai
```

Ollama needs no key. Because the model defaults are Anthropic model IDs,
`FURROW_PROVIDER=ollama` also requires setting `FURROW_MODEL` (and the role
models) to names your local server serves.

## How it works

Each cycle runs three roles:

1. **Planner** (`furrow/agents/planner.py`) turns the goal into a plan: a list
   of tasks, each with an id, description, target files, and dependencies, plus
   a rationale. JSON is extracted leniently (bare, fenced, or embedded in prose)
   and normalized; a reply that yields no usable task is re-asked, up to
   `MAX_PLAN_ATTEMPTS` (3) times.
2. **Workers** (`furrow/agents/worker.py`) run one task each, keeping at most
   `FURROW_MAX_PARALLEL_TASKS` in flight. A task starts only once every task it
   lists in `dependencies` has completed; if one of those failed or was skipped,
   the task is marked skipped instead. A worker is given the task's current file
   contents (budgeted) plus a bounded workspace listing, and writes files from
   its reply using a plain-text block protocol — no tool calling, no JSON:

   ```text
   <<<FILE: src/auth.py>>>
   ...the complete new contents of the file...
   <<<END FILE>>>

   <<<EDIT: src/auth.py>>>
   <<<SEARCH>>>
   the exact contiguous text to find, copied from the file
   <<<REPLACE>>>
   the text that replaces it
   <<<END EDIT>>>
   ```

   A `FILE` block creates a file or rewrites it whole; an `EDIT` block replaces
   an exact snippet, and if that snippet is missing or ambiguous the edit is
   skipped and reported, never guessed. Paths are resolved against the
   workspace, and any path that escapes it — absolute, `~`, or resolving through
   `..` or a symlink — is refused, as are directories and the workspace root
   itself. Refused and skipped changes are collected into the task's result
   rather than raised.
3. **Tester** (`furrow/agents/tester.py`) picks a runner from the project's
   manifests (`pytest` for Python, `npm`/`pnpm`/`yarn`, `cargo`, `go`), runs it
   in the workspace under `FURROW_TEST_TIMEOUT`, and decides pass/fail from the
   process **exit code** — never from the output. If no runner is detected, the
   cycle fails with "No test runner detected" and the LLM is not consulted at
   all. The LLM is only asked to phrase that verdict, and cannot overrule it.

If tests fail — or a task fails — the failures become the focus for the next
cycle, so the loop plans a fix and tries again.

The loop halts for exactly one of these reasons, reported as `stopped_reason`:

| Reason | Meaning |
| --- | --- |
| `goal-complete` | A cycle's tests passed and no task failed |
| `no-tasks` | The planner returned an empty plan |
| `max-cycles` | `FURROW_MAX_CYCLES` was reached (only when non-zero) |
| `stalled` | Three consecutive cycles repeated the same state — the same test failures, the same failing tasks, three planner failures, or three crashed cycles. The failures are still reported; the point is to stop burning API credits rather than spin forever |
| `cancelled` | The run was cancelled — in the web UI, by the browser closing the websocket |

Only `goal-complete` exits `0`; every other reason reports failure.

Output is rendered with `rich` on the terminal; `FURROW_LOG_LEVEL` controls
library-level logging.

## Development

```bash
pytest            # test suite
ruff check .      # lint
mypy furrow       # types
```