# Furrow

Furrow is an autonomous coding agent. You give it a goal; it repeatedly plans a
set of tasks, executes them in parallel through sandboxed file tools, runs your
test suite to verify the result, and feeds any failures back into the next round
of planning until the goal is met.

The loop is bounded. It stops on exactly one of four conditions:

1. **Stop requested** — `request_stop()`, from Ctrl-C or the web UI's Stop button.
2. **Cycle budget spent** — `max_cycles` is reached. The default `0` means *no
   cycle limit*, not "forever"; conditions 1, 3 and 4 still apply.
3. **The planner returns an empty task list** — the model's signal that the goal
   is met and there is nothing left to do.
4. **Stall** — `max(3, max_parallel_tasks)` consecutive cycles complete without
   a single finished task. A bad goal tends to fail the same way every round, so
   rather than spend the whole budget repeating itself, the loop gives up.

## Requirements

- Python 3.10+
- An API key for one of the supported providers: Anthropic, OpenAI, or a local
  Ollama server

## Install

Furrow is **not published on PyPI** — `pip install furrow` does not work. Install
it from a local checkout.

With no install step at all, run the module straight from the source tree:

```bash
PYTHONPATH=$PWD .venv/bin/python -m furrow --help
```

Or install the checkout in editable mode, which also puts a `furrow` console
script on your path:

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

```bash
pip show furrow        # Package(s) not found — nothing is installed until you do this
```

## Usage

### `start` — work a goal from the terminal

```bash
# Prompt for the goal interactively
PYTHONPATH=$PWD .venv/bin/python -m furrow start

# Or pass it directly
PYTHONPATH=$PWD .venv/bin/python -m furrow start "Add JWT auth to the API"

# Bound the run and tell it how to verify its own work
PYTHONPATH=$PWD .venv/bin/python -m furrow start "Add JWT auth" \
    --test-command "pytest -q" \
    --max-cycles 12 \
    --max-parallel-tasks 4

# Run against a local Ollama
PYTHONPATH=$PWD .venv/bin/python -m furrow start "Fix the flaky test" \
    --provider ollama \
    --model llama3.1

# Restrict the blast radius to a subtree, and cap worker tool-calling
PYTHONPATH=$PWD .venv/bin/python -m furrow start "Document the parser" \
    --workspace ./furrow \
    --worker-iterations 8 \
    --verbose
```

`--model` sets the model for **all four** roles at once; `--planner-model`,
`--worker-model` and `--tester-model` each override `--model` for a single role.

A `FurrowError` is reported as a single actionable line with exit code `2`
rather than a traceback. Ctrl-C exits `130` after a graceful shutdown.

### `web` — the browser UI

```bash
PYTHONPATH=$PWD .venv/bin/python -m furrow web
# → http://127.0.0.1:8000

PYTHONPATH=$PWD .venv/bin/python -m furrow web --host 127.0.0.1 --port 9000
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | The dashboard |
| `/health` | GET | Liveness check |
| `/api/status` | GET | Current loop state |
| `/api/start` | POST | Start a run; `409` if one is already in flight |
| `/ws` | WS | Live event stream |

### `config` — show what is actually resolved

```bash
PYTHONPATH=$PWD .venv/bin/python -m furrow config
```

```
provider               anthropic
model                  claude-sonnet-4-20250514
planner_model          claude-3-5-haiku-20241022
worker_model           claude-3-5-sonnet-20241022
tester_model           claude-3-5-sonnet-20241022
anthropic_api_key      not set
openai_api_key         not set
ollama_base_url        http://localhost:11434
max_parallel_tasks     5
max_cycles             0
workspace              /path/to/checkout
log_level              INFO
test_command           None
test_timeout           120
worker_max_iterations  12
allow_shell            False
shell_timeout          60
log_json               False
warning: provider is anthropic but no API key configured
```

Secrets are masked (a set key prints as `set (****abcd)`). Anything
`Settings.validate_runtime()` objects to is printed as a `warning:` line — it is
advisory and never raises, so the command still exits `0`.

## Configuration

Every field on `Settings` is read from an environment variable with the
`FURROW_` prefix, or from a `.env` file in the working directory. Precedence,
highest first:

1. CLI flag
2. `FURROW_*` environment variable
3. `.env`
4. Built-in default

**[.env.example](.env.example) is the reference** — it documents all 18 settings
with their types, defaults and constraints. Copy it to start:

```bash
cp .env.example .env
```

`FURROW_LOG_JSON` is declared on `Settings` but read nowhere in the code, so
setting it currently does nothing.

### API keys

Use the `FURROW_`-prefixed variables:

```bash
FURROW_ANTHROPIC_API_KEY=sk-ant-...
FURROW_OPENAI_API_KEY=sk-...
```

As a fallback only, `furrow/llm.py` also reads the conventional unprefixed
`ANTHROPIC_API_KEY` and `OPENAI_API_KEY`, and only when the corresponding
`FURROW_*_API_KEY` is unset. Prefer the prefixed form so the resolution order
stays predictable.

### Web server token

`FURROW_WEB_TOKEN` is read straight from the environment by `furrow/web/server.py`
— it is deliberately *not* one of the 18 `Settings` fields and is therefore not
in `.env.example`.

## Architecture

Four layers, each depending only on the ones above it:

```
furrow/config.py          pydantic models + Settings (all FURROW_* resolution)
        ↓
furrow/llm.py             provider adapter: Anthropic / OpenAI / Ollama, retries, usage
        ↓
furrow/agents/            planner.py · worker.py · tester.py (+ prompts.py)
        ↓
furrow/core/orchestrator.py   the plan → execute → verify loop
```

Two front-ends drive the same orchestrator:

```
furrow/cli/               main.py — click commands: start, web, config
furrow/web/               server.py — FastAPI + WebSocket; dashboard.py — the UI
```

Supporting modules:

- **`furrow/errors.py`** — the exception taxonomy, rooted at `FurrowError`. The
  CLI catches this base class and turns it into a message plus exit code 2.
- **`furrow/parsing.py`** — pulls JSON out of whatever the model actually
  returned: markdown fences, prose around the object, brace-balanced scanning
  that is aware of string literals and escape sequences.
- **`furrow/tools.py`** — the five workspace tools and their sandbox.
- **`furrow/core/events.py`** — `EventType` (12 members), `Event`, and `EventBus`.
  Handlers may be sync or async; dispatch iterates a snapshot so a handler can
  subscribe or unsubscribe mid-dispatch; a handler that raises is swallowed so
  one bad subscriber cannot kill the loop.
- **`furrow/core/reporting.py`** — `RichReporter`, an `EventBus` subscriber that
  renders events to the terminal.

That last split is the reason the web UI can stream live progress: the loop
emits events and knows nothing about how they are displayed. The terminal
renderer and the WebSocket bridge are just two subscribers to the same bus.

## How a cycle works

1. **Plan.** The planner turns the goal into a task list. Tasks carry real
   `dependencies` (hard ordering constraints) and a `files` list (soft mutual
   exclusion — two tasks that both claim the same file will not run at the same
   time, but two tasks that both declare `files=[]` *are* parallelised).
2. **Batch.** The orchestrator groups tasks into batches that respect those
   constraints. An `asyncio.Semaphore(max_parallel_tasks)` caps how many workers
   are in flight.
3. **Execute.** Each worker runs a real agentic tool loop, capped at
   `worker_max_iterations` (default 12) rounds of tool calls. Workers genuinely
   modify files — this is not a chat transcript that describes changes, it is
   the thing making them.
4. **Verify.** The tester runs `test_command` as a subprocess and judges the
   result. **The subprocess exit code outranks the model's verdict**: a model
   that reports `passed: true` after a non-zero exit is vetoed. A failing verdict
   always carries a non-empty `failures` list.
5. **Feed back.** Those failures become repair context on the *next* cycle's
   goal. The original goal is never overwritten — the context is appended and
   capped at 4,000 characters. (That non-empty guarantee matters: an empty
   failure list here would leave the next cycle with nothing to do and the loop
   would exit early having accomplished nothing.)

## Safety

**Workspace sandbox.** Every tool path is resolved and then checked with
`relative_to(root)` against the workspace root, so `../../etc/passwd` and
symlink escapes are rejected before any I/O. Writes are atomic — content goes to
a temp sibling and is then `os.replace`d into position, so a crashed worker
cannot leave a half-written file. Tool output is truncated at 20,000 characters
and `list_files` caps at 500 entries, which keeps a runaway agent from blowing up
its own context window.

**`allow_shell` is off by default.** With it off, `run_command` is refused
outright and the agent can only touch files. Turning it on lets the agent execute
**arbitrary shell commands inside the workspace** — which is arbitrary code
execution with your privileges. It is gated behind a confirmation prompt, and in
scripts or CI it additionally requires `--yes` or `--dangerously-allow-shell`.
Enable it only in a disposable sandbox.

**Web server binds `127.0.0.1:8000` by default**, so it is not reachable from
other machines out of the box. If `FURROW_WEB_TOKEN` is set, every endpoint is
gated: `/health`, `/api/status` and `/api/start` accept **only**
`Authorization: Bearer`, while `/` and `/ws` additionally accept `?token=` —
a browser cannot set headers on a page load or on a WebSocket handshake, so
the dashboard cannot use the header. With no token set the server is
unauthenticated and logs a prominent warning; binding a non-loopback address
without a token also warns, but still binds.
The dashboard in `furrow/web/dashboard.py` is self-contained and XSS-safe — no
`innerHTML` is ever assigned untrusted data.

## Development

```bash
.venv/bin/python -m pytest -q
```

```
356 passed in 7.55s
```

```bash
.venv/bin/python -m ruff check .
```

```
All checks passed!
```

Tests need no API key and make no network calls. pytest is configured in
`pyproject.toml` with `asyncio_mode = "auto"` and `pythonpath = ["."]`, so it
runs from a clean checkout without installing the package.
