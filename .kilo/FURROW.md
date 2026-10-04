# Furrow Architecture

## Overview

Furrow is an autonomous coding agent, written in Python, that takes a high-level goal
and drives it to completion in bounded cycles: plan the goal into parallelisable
tasks, execute the tasks concurrently through tool-using worker agents, run the
project's own test command over the result, and then either start another cycle or
stop.

The loop is **not** infinite. It terminates on exactly four conditions, described
under [Orchestrator](#1-orchestrator-furrowcoreorchestratorpy). In practice it usually
terminates because the planner eventually decides the goal is done and returns an
empty task list.

Furrow is two products' worth of code sharing one name, and this document covers both:

- **The Python package** (`furrow/`) — the actual product. Run it with
  `python -m furrow`. This is what everything below describes.
- **The Kilo-side agent definitions** (`.kilo/agent/*.md`, `.kilo/command/furrow.md`)
  — prompt text that makes *Kilo* delegate to its own subagents. It is a separate
  system that neither imports nor is imported by the Python package. It has drifted
  from the Python implementation; see
  [Prompt drift](#prompt-drift-between-kiloagentmd-and-furrowagentspy).

The package is four layers deep, with two front-ends on top:

```
config.py        models + Settings          (18 settings fields, FURROW_* env prefix)
   ↓
llm.py           provider adapter           (Anthropic / OpenAI / Ollama, retries, tool loop)
   ↓
agents/          Planner, Worker, Tester    (prompting + response parsing)
   ↓
core/orchestrator.py  the loop               (batching, scheduling, termination)
   ↑
cli/main.py  and  web/server.py             (two front-ends over the same core)
```

Two modules cut across all of it: `tools.py` (the workspace sandbox every agent
writes through) and `core/events.py` (the bus every layer publishes progress on).

## Components

### 1. Orchestrator (`furrow/core/orchestrator.py`)

The loop itself. One cycle is:

1. **Goal** — the run starts from a single original goal, held in `_original_goal`.
2. **Plan** — `Planner.plan()` turns the goal plus accumulated repair context into a
   `Plan` of tasks. `plan.created` is published.
3. **Batch and execute** — tasks are grouped into dependency-legal batches (see
   [Tool sandbox and scheduling](#tool-sandbox-and-scheduling)) and run concurrently
   under an `asyncio.Semaphore(max_parallel_tasks)`.
4. **Test** — the `Tester` runs `Settings.test_command`; the verdict is published as
   `tests.completed`.
5. **Decide** — the four stop conditions are evaluated; if none fires, repair context
   and per-task feedback are folded into the next cycle's goal and the loop repeats.

Context is deliberately bounded so a long run cannot grow the goal without limit:

- The **original goal is never overwritten** — cycle 2 works on the same goal plus
  context, not a mutated replacement.
- **Repair context is capped at 4,000 characters.**
- **Per-task feedback is capped at 2,000 characters** before being handed to a worker.

`request_stop()` is cooperative: it asks the loop to finish *after the current cycle*
rather than cancelling in-flight workers, so a partially-written file is never left
behind by a mid-task kill.

**The four termination conditions — the complete list:**

| # | Condition | Detail |
|---|-----------|--------|
| 1 | `request_stop()` was called | Ctrl-C in the CLI, or the Stop button / `{action: "stop"}` frame on the web socket. A non-empty reason is what `run.completed` reports. |
| 2 | `max_cycles` reached | `Settings.max_cycles`, default `0` meaning *no cycle limit*. |
| 3 | The planner returned an **empty task list** | The planner's way of saying the goal is satisfied. |
| 4 | **Stall threshold** | `max(3, max_parallel_tasks)` consecutive cycles that complete no new task and never get a passing test run. |

`max_cycles = 0` is what makes an *unbounded* run possible, but not an immortal one —
conditions 3 and 4 always remain. A run that stops on condition 4 says so explicitly in
its `run.completed` reason (`stalled: N consecutive cycle(s) ...`), so a stalled run is
distinguishable from a finished one.

### 2. Planner (`furrow/agents/planner.py`, prompts in `furrow/agents/prompts.py`)

Breaks the goal into parallelisable tasks. Output is a `Plan` with `tasks[]`, each
task carrying `id`, `description`, `files` and `dependencies`.

Model output is not trusted to be clean JSON: `furrow/parsing.py` extracts JSON from
whatever the model returns — markdown fences, JSON embedded in surrounding prose, or a
balanced-brace scan — and raises `StructuredOutputError` when there is nothing
extractable.

Returning `tasks: []` is the planner's stop signal (termination condition 3), so an
empty list is load-bearing behaviour, not a degenerate case.

### 3. Worker (`furrow/agents/worker.py`)

Executes exactly one task. Workers are genuinely agentic: the worker runs a real tool
loop — think, call a `WorkspaceTools` method, read the result, decide again — bounded
by `Settings.worker_max_iterations`, **default 12**. Workers therefore actually modify
files; this is not a single-shot prompt.

The worker holds a `WorkerResult` (success flag, summary, error, per-task feedback),
which the orchestrator records as `task.completed` / `task.failed`.

### 4. Tester (`furrow/agents/tester.py`)

Runs `Settings.test_command` and reports a verdict. Two invariants make it trustworthy:

- **The exit code outranks the model.** A parseable `{"passed": true}` cannot survive a
  non-zero exit code; `passed` is rewritten to `False` and the run's own output is
  used instead.
- **A failing verdict always carries non-empty `failures`.** The orchestrator feeds
  `"\n".join(failures)` into the *next cycle's goal*, so a `passed: false` with an
  empty (or whitespace-only) failure list would leave the next cycle with nothing to
  work on. Missing failures are synthesised from the exit code and captured output.

The Python tester **observes and reports; it never repairs.** It holds no tools and
writes nothing. Repair is the next cycle's planner problem, by design — which is why a
cycle boundary is the natural repair point.

### 5. Command (`/furrow`, `.kilo/command/furrow.md`)

The Kilo-side slash command. It invokes the `furrow` Kilo agent with a goal from
`$ARGUMENTS`. It drives the *Kilo* agent hierarchy, not `python -m furrow`.

### Prompt drift between `.kilo/agent/*.md` and `furrow/agents/*.py`

The markdown agents and the Python agents are independent implementations of the same
idea, and they no longer agree. Read the markdown as the Kilo-side prompt text, not as
a description of the Python code:

- **Iteration bound.** `.kilo/agent/furrow.md` still tells the planner that "each task
  should be completable in 1-3 tool call rounds" (that claim was also copied into
  earlier versions of this document). The Python worker has no such notion: it runs a
  tool loop bounded by `worker_max_iterations`, default 12. `.kilo/agent/furrow-worker.md`
  sets only a `steps: 15` front-matter budget for the Kilo subagent, which is a
  different mechanism again.
- **Who fixes failures.** `.kilo/agent/furrow-tester.md` says "if tests fail, diagnose
  and fix them" and "only fix what is needed to make tests pass", and
  `.kilo/agent/furrow.md` step 4 says "fix failures". `furrow/agents/tester.py` does
  none of that — it reports a verdict and leaves repair to the next planned cycle.
- **Termination.** `.kilo/agent/furrow.md` still describes "an infinite development
  loop" that "runs forever" until the user stops it. The Python loop has the four
  conditions above.

The Python prompt text lives separately in `furrow/agents/prompts.py`; nothing keeps
the two prompt sets in sync.

## Event bus and reporting

This is the most important architectural idea in the codebase.

**The problem it solves.** The loop originally printed progress straight to a global
`rich.Console`. That works for exactly one consumer — a terminal — and leaves no seam
for anything else. There was no way for the web server, a log file, or a test to observe
the loop without monkey-patching the console.

**The bus is that seam.** `furrow/core/events.py` defines:

- `EventType` — 12 members: `run.started`, `run.completed`, `run.failed`,
  `cycle.started`, `plan.created`, `task.started`, `task.completed`, `task.failed`,
  `task.skipped`, `tests.completed`, `cycle.completed`, `log`.
- `Event` — a slotted dataclass of `type`, `data`, `cycle`, `timestamp`, with
  `to_dict()` for JSON-safe consumption. `data` values must be JSON-safe; that is the
  emitter's responsibility.
- `EventBus` — fan-out. Handlers may be **sync or async** (the bus calls the handler
  and awaits the result if it is awaitable, so an `async def __call__` object works as
  well as a coroutine function). Iteration is over a **snapshot** of the handler list,
  so a handler may subscribe or unsubscribe during dispatch. **Per-handler exceptions
  are caught, logged and swallowed** — one broken subscriber must never be able to kill
  the run.

`furrow/core/reporting.py` provides `RichReporter`, a bus subscriber that renders cycle
panels to the terminal. It is the *default* consumer, not a hard-coded one.

This is what makes the web dashboard possible at all. `furrow/web/server.py` subscribes
its own handler to the same bus and bridges events onto an `asyncio.Queue`, which a
WebSocket task drains and sends as JSON. The orchestrator never knows a browser is
attached; it only publishes. The CLI and the web server render the identical event
stream through completely different consumers — which is the entire payoff of the bus.

## Tool sandbox and scheduling

### The sandbox (`furrow/tools.py`)

`WorkspaceTools` exposes exactly five tools: `read_file`, `write_file`, `edit_file`,
`list_files`, `run_command`. `specs()` returns the neutral definitions;
`anthropic_tools()` / `openai_tools()` render them in each provider's wire format.

- **Path containment.** Every path goes through `_resolve()`, which resolves it to an
  absolute path and then requires `relative_to(root)`. Anything that lands outside the
  workspace — `..` traversal, absolute paths, symlinks pointing out — raises
  `WorkspaceContainmentError`. Containment is checked on *every* tool call, not once at
  construction.
- **Atomic writes.** `write_file` writes to a temporary file in the destination
  directory and then `os.replace`s it into position, so an interrupted or failed write
  cannot leave a half-written source file behind.
- **Output truncation.** Tool output is truncated at `max_output_chars`, default
  20,000, with an explicit notice carrying how much was dropped — so the model is never
  silently starved of context.
- **The shell gate.** `run_command` is refused outright unless `allow_shell=True`.
  `allow_shell` defaults to `False` in `Settings`, and the CLI additionally requires
  `--yes` or `--dangerously-allow-shell` to turn it on.

### The scheduler

Tasks are grouped into batches that are safe to run concurrently:

- **`dependencies` are hard ordering constraints.** A task never starts before every
  task it names has finished.
- **Shared `files` are soft mutual exclusion.** Two tasks that list the same file are
  serialised against each other, because they would otherwise race on the same write.
- **Two tasks that both declare `files = []` are *not* serialised.** A task that
  declares no files is declaring no conflict, and treating it as conflicting with every
  other empty-`files` task would collapse the whole plan into a sequential run.

Concurrency is then capped by `asyncio.Semaphore(max_parallel_tasks)` (default 5), so
batch width and total in-flight work are bounded independently.

## Files

```
furrow/
  __init__.py           public surface re-exported (Orchestrator, EventBus, LLMClient, ...)
  __main__.py           enables `python -m furrow`
  config.py             Plan / TaskModel / TestResult models + 18-field Settings
  errors.py             exception taxonomy rooted at FurrowError
  llm.py                LLMClient: Anthropic / OpenAI / Ollama adapter, retries, tool loop
  parsing.py            robust JSON extraction from messy model output
  tools.py              WorkspaceTools: the 5-tool workspace sandbox
  agents/
    __init__.py
    planner.py          goal -> Plan
    prompts.py          the Python-side prompt templates
    tester.py           test command -> verdict (exit code wins)
    worker.py           agentic tool loop for one task
  cli/
    __init__.py
    main.py             start / web / config + --version
  core/
    __init__.py
    events.py           EventType, Event, EventBus
    orchestrator.py     the loop, batching, termination
    reporting.py        RichReporter (bus subscriber)
  web/
    __init__.py
    dashboard.py        XSS-safe single-page dashboard
    server.py           FastAPI app: REST + WebSocket event bridge
tests/
  __init__.py
  conftest.py           shared fixtures (offline fakes, env isolation)
  test_agents_planner.py
  test_agents_tester.py
  test_agents_worker.py
  test_cli.py
  test_config.py
  test_events.py
  test_llm.py
  test_orchestrator.py
  test_parsing.py
  test_tools.py
.kilo/
  FURROW.md             this document
  agent/                furrow, furrow-planner, furrow-worker, furrow-tester (Kilo-side)
  command/furrow.md     /furrow slash command (Kilo-side)
.env.example            tracked template for FURROW_* settings
.gitignore              excludes .venv/, caches, .env (but not .env.example)
kilo.json               Kilo project config: default agent + open permissions
pyproject.toml          setuptools build, deps, pytest/mypy/ruff config
README.md               user-facing documentation
```

There is no `LICENSE` file in the repository.

## Safety

**Workspace containment.** All file access from an agent goes through `WorkspaceTools`,
which resolves and then confines every path to the workspace root. Symlinks, `..` and
absolute paths cannot escape it.

**Shell is opt-in.** `run_command` is refused unless `allow_shell=True`; the setting
defaults to `False`, and the CLI requires an explicit `--yes` /
`--dangerously-allow-shell` acknowledgement to enable it.

**Writes are atomic and outputs are bounded**, so a killed run cannot leave a truncated
source file, and no single tool result can flood the context window.

**Web auth posture.** The server binds `127.0.0.1:8000` by default. Setting
`FURROW_WEB_TOKEN` gates:

| Endpoint | Auth accepted |
|----------|---------------|
| `GET /` | `Authorization: Bearer <token>` |
| `GET /health` | `Authorization: Bearer <token>` |
| `GET /api/status` | `Authorization: Bearer <token>` |
| `POST /api/start` | `Authorization: Bearer <token>` |
| `WS /ws` | `?token=<token>` query parameter |

With no token set the server is unauthenticated (it logs a warning saying so), which is
acceptable only because of the loopback default; the CLI *warns* when asked to bind a
non-loopback host without `FURROW_WEB_TOKEN`, but it binds anyway — the warning is not
a refusal. `POST /api/start` also returns **409** if a run is
already active, so a runaway orchestrator cannot stack up runs.

**`FURROW_WEB_TOKEN` is read from `os.environ` at import time**, not as a `Settings`
field. It is compared with `==`, not `compare_digest`, and changing it requires
restarting the process. Anyone expecting it to behave like the other `FURROW_*`
settings should know it does not.

## Permissions

`kilo.json` sets `bash: allow`, `edit: allow`, `read: allow`, and
`default_agent: "furrow"`.

These permissions govern the **Kilo-side** agents — what a `furrow-worker` subagent
spawned by Kilo may do. They are **entirely separate from Furrow the product's own
sandbox**: `WorkspaceTools` enforces path containment and the shell gate at runtime
regardless of what `kilo.json` says. Opening up `kilo.json` does not open up the Python
package, and the sandbox does not constrain Kilo's own tool calls.

## Usage

Furrow is **not published on PyPI**. `pip install furrow` will not fetch this project.

Two invocations that actually work:

```bash
# 1. From the repo root — the root is on sys.path, so this just works
python -m furrow start "add pagination to the /api/items endpoint"

# 2. From anywhere, with the repo on PYTHONPATH
PYTHONPATH=/path/to/furrow python -m furrow start "add pagination to /api/items"

# 3. Editable install — also provides the `furrow` console script
python -m pip install -e .
furrow start "add pagination to /api/items"
```

(An editable install only works because the build backend is pinned to
`setuptools.build_meta`; it was previously invalid and the package could not be
installed at all.)

Commands:

```bash
python -m furrow start [GOAL]   # run the loop
python -m furrow web             # serve the dashboard (127.0.0.1:8000)
python -m furrow config          # print resolved settings, secrets masked
python -m furrow --version
```

Settings precedence is **CLI flag > `FURROW_*` environment variable > `.env` > default**.
`--model` sets all four model fields (main, planner, worker, tester) at once;
`--planner-model` / `--worker-model` / `--tester-model` override individual roles.

Exit codes: `0` on success, `2` for any `FurrowError`, `130` on Ctrl-C.

`--allow-shell` will not enable command execution on its own — it also needs
`--yes` or `--dangerously-allow-shell`.

## Future Work

Genuinely open, in rough priority order:

- **Persistent state file for goal/task tracking across sessions.** There is no state
  file anywhere in the code — nothing in `furrow/` writes or reads one, so a run's
  history dies with the process and a restart re-plans from scratch.
- **No git awareness whatsoever.** Nothing in `furrow/` shells out to git. A run
  mutates the working tree with no branch, no checkpoint and no rollback, and no diff
  in the event stream. A per-cycle branch or worktree plus a diff summary on
  `task.completed` would make runs reviewable and recoverable.
- **Stopping does not cancel in-flight work.** `request_stop()` is cooperative: it lets
  the current cycle finish. That is the safe choice for file writes, but a long worker
  turn cannot be interrupted, and the web Stop button does not stop a run promptly.
- **No token streaming.** `llm.py` never requests `stream=True`, so a long worker turn
  produces no output until it completes; the UI only moves at task boundaries.
  Streaming into the bus would make long turns observable.
- **No cost or token budget.** Per-call `Usage` is recorded in `llm.py` but never
  aggregated across a run, and `Settings` has no spend cap, so a long run's cost is
  unbounded and invisible.
- **Duplicated, drifted prompt text.** The Kilo markdown agents and
  `furrow/agents/prompts.py` describe the same three roles separately and disagree (see
  [the drift note](#prompt-drift-between-kiloagentmd-and-furrowagentspy)). Generating
  the markdown from one source, or deleting one of the two hierarchies, would remove a
  permanent source of false documentation.
- **Ollama only via the OpenAI-compatible path.** `LLMClient` serves `Provider.OLLAMA`
  through the OpenAI client pointed at `ollama_base_url`; there is no native Ollama
  client, so provider-specific capabilities are unavailable for local models.
