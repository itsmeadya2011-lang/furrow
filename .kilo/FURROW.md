# Furrow Agent Architecture

Two things share the name "Furrow":

- The **Kilo agent suite** under `.kilo/` — the orchestrator agent plus its
  planner, worker and tester subagents, entered through `/furrow`. This document
  describes that suite and how it relates to the installed Python package.
- The **Python package** in `furrow/`, which implements the same
  plan/execute/test loop for the `furrow` CLI and the `furrow web` UI.

Both exist. The Kilo agents are the primary product surface; the package is the
library and CLI the same loop runs headless.

## Overview

Furrow runs a **bounded** development loop for web/CLI/desktop projects. Each
cycle plans parallel tasks, executes them, runs the test suite, and folds
failures back into the next round.

Despite the "infinite loop" branding, the loop always halts:

- `max_cycles` caps the number of cycles (`0`, the default, means unlimited).
- `MAX_UNPRODUCTIVE_CYCLES = 3` consecutive cycles that complete no previously
  unsettled task stop a self-repair loop that is going nowhere.
- `MAX_REPAIR_ATTEMPTS = 3` consecutive repair cycles that leave the suite
  failing stop a loop that cannot fix a permanently red suite.

In the Python implementation these two guards are class attributes on
`Orchestrator`. In the Kilo orchestrator agent they are the stopping rules in
[`.kilo/agent/furrow.md`](agent/furrow.md). Neither flavour runs forever on its
own.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent); `default_agent` in `kilo.json`
- **Role:** Runs the loop. Receives the goal, plans, delegates, tests, repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5, capped by `MAX_TASKS`)
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Fold failures into the next cycle's goal; report and repeat
- **Stop conditions:** goal complete, cycle budget exhausted, either stall guard
  tripped, or the user stopped the session. It does **not** ask "should I
  continue?" between cycles and it does **not** require the goal to be complete
  before halting.

### 2. Planner (`furrow-planner`)
- **Type:** Subagent (`PlannerAgent` in `furrow/agents/planner.py`)
- **Role:** Breaks a high-level goal into parallelizable tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`,
  `dependencies` and `status`, plus a `rationale`.
- **Enforcement:** one repair request is made when the reply is malformed JSON.
  The task-count limit is not left to the prompt — the planner truncates
  over-long plans, drops dependency ids that name a missing or self-referential
  task, normalises statuses, and appends a note to `rationale` whenever it
  changed something. An empty plan raises `ValueError`, which the orchestrator
  records and recovers from.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent (`WorkerAgent` in `furrow/agents/worker.py`)
- **Role:** Implements a single assigned task **and writes the changed files to
  disk**. It is the only stage that mutates the workspace.
- **Protocol:** the model replies with
  `{"summary": "...", "files": [{"path": "...", "content": "..."}]}` and each
  `content` is the complete new text of the file, never a diff.
- **Sandbox:** every write goes through `LLMClient.write_file`, which resolves
  the path against `FURROW_WORKSPACE` and refuses anything that escapes it
  (absolute paths, `../` segments, symlinks). Missing parent directories are
  created.
- **Failure handling:** a reply that is not valid JSON degrades to a plain
  summary with zero files written rather than failing the task; individual
  write failures are collected in `WorkerAgent.errors` instead of aborting the
  remaining files. Paths actually written are collected in
  `WorkerAgent.files_written` and forwarded on the task-complete event.
- **Constraint:** a task is expected to need 1-3 work rounds.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent (`TesterAgent` in `furrow/agents/tester.py`)
- **Role:** Runs the workspace test suite and judges the outcome.
- **Command detection:** `FURROW_TEST_COMMAND`/`--test-command` wins; otherwise
  `pytest`, `python -m pytest`, `npm test`, `pnpm test`, `yarn test`,
  `cargo test` and `go test ./...` are probed with `shutil.which` and the first
  available one runs. No runner found is itself a failure, not a silent pass.
- **Authoritative signal:** the process is the hard signal. A `passed: true`
  verdict from the model is **overruled** when the command exited non-zero,
  timed out or could not run; the result is downgraded to failing and the
  override is stated in the summary. When the model's reply is unparsable the
  verdict is derived from the exit status alone. The model supplies *why*,
  which becomes the next cycle's goal; it never gets to claim a green suite.

### 5. Observability layer (`furrow/core/events.py`)
- **Type:** Library component, no Kilo agent
- **Role:** The only usable interface for anything other than a terminal, since
  a run lasts arbitrarily long.
- **Contents:** `EventType` (12 stable wire strings: `run_start`, `cycle_start`,
  `plan`, `task_start`, `task_complete`, `task_failed`, `task_skipped`,
  `test_start`, `test_result`, `cycle_end`, `error`, `run_end`),
  `FurrowEvent` (type, message, cycle, task_id, data) and `RunResult`
  (cycles, completed, tasks_completed, tasks_failed, cycles_failed, last_error,
  stopped_reason).
- **Delivery:** the orchestrator publishes each event through a synchronous
  `on_event` callback; a listener that raises is ignored so it cannot break the
  run. The CLI renders them with `rich`, and `furrow web` forwards them to the
  browser as JSON frames.

### 6. Command (`/furrow`)
- **Type:** Slash command (`.kilo/command/furrow.md`)
- **Role:** Entry point. Invokes the `furrow` orchestrator agent with a goal from
  `$ARGUMENTS`.

### 7. CLI and web UI (`furrow/cli/main.py`, `furrow/web/server.py`)
- **Type:** Library components
- **Role:** Non-TUI entry points to the same loop. `furrow start` runs it in the
  terminal and maps the outcome onto a meaningful exit code (0 complete, 1 ran
  but did not complete, 2 bad configuration, 130 interrupted). `furrow web`
  serves the streaming UI.

## Task-size limits and the single source of truth

Two limits appear in this suite as prose and must not drift:

| Limit | Value |
| --- | --- |
| Tasks per plan | `MIN_TASKS = 1` to `MAX_TASKS = 5` |
| Work rounds per task | `MIN_TASK_ROUNDS = 1` to `MAX_TASK_ROUNDS = 3` |

**The single source of truth is `furrow/agents/prompts.py`**, which defines
`MIN_TASKS`, `MAX_TASKS`, `MIN_TASK_ROUNDS` and `MAX_TASK_ROUNDS` as module
constants and interpolates them into the prompt templates.

`MAX_TASKS` is **also enforced in code** by `PlannerAgent._enforce_limits`: an
over-long plan is truncated to `MAX_TASKS` and a note is appended to `rationale`
recording how many tasks were dropped, so truncation is never silent. An empty
plan raises `ValueError`.

Where this document, `.kilo/agent/furrow.md` and
`.kilo/agent/furrow-planner.md` restate these numbers, `prompts.py` wins.

## Files

```
.kilo/
  agent/
    furrow.md          # Orchestrator agent (bounded loop)
    furrow-planner.md  # Planner subagent
    furrow-worker.md   # Worker subagent
    furrow-tester.md   # Tester subagent
  command/
    furrow.md          # /furrow slash command
kilo.json              # Project config (default agent = furrow, open permissions)

furrow/
  __init__.py          # Public API re-exports and __version__
  config.py            # Provider, TaskStatus, TaskModel, Plan, TestResult, Settings
  llm.py               # LLMClient: provider facade, retry/backoff, sandboxed file I/O
  py.typed             # PEP 561 marker (typed package)
  agents/
    __init__.py
    prompts.py         # Prompt templates and the MIN/MAX task limits
    parsing.py         # strip_code_fence, extract_json, parse_json_response, coerce_status
    planner.py         # PlannerAgent
    worker.py          # WorkerAgent, WorkerResponse, FileEdit
    tester.py          # TesterAgent, CommandResult
  core/
    __init__.py
    events.py          # EventType, FurrowEvent, RunResult
    orchestrator.py    # Orchestrator: the bounded plan/execute/test loop
  cli/
    __init__.py        # present so the `furrow` console script ships in the wheel
    main.py            # click CLI: `start` and `web`, exit-code mapping
  web/
    __init__.py
    server.py          # FastAPI app: /, /health, POST /api/start, /ws
tests/                 # pytest suite (pytest-asyncio auto mode)
.env.example           # Copy to .env; every FURROW_* setting, all commented out
.gitignore
pyproject.toml         # setuptools.build_meta, console script, ruff/mypy/pytest config
README.md
```

## Permissions

Open by default: `bash: allow`, `edit: allow`, `read: allow`.

This is appropriate for the Kilo suite, which operates on the user's own
repository under their supervision. It is **not** appropriate for `furrow web`:
that server has no authentication, which is why it binds `127.0.0.1` by default
and prints a loud warning when asked to bind anything else.

## Usage

- Run `/furrow <goal>` to start the loop in the TUI.
- The agent runs until the goal is complete, one of its three bounds trips (see
  the stopping rules in [`agent/furrow.md`](agent/furrow.md)), or the user stops
  the session.
- Mid-loop user input is incorporated into the next cycle's plan.
- Headless: `furrow start "<goal>" --max-cycles 5`, then read the exit code.
- Watching: `furrow web`, then open `http://127.0.0.1:8000`.

## Future Work

- On-disk persistence for goal/task tracking. The Python orchestrator keeps a
  task registry that accumulates across cycles *within one run*, but nothing is
  written to disk: a run that ends cannot be resumed, and the Kilo suite keeps
  its state only as agent context.
- Authentication and authorisation for `furrow web`. It is currently a
  loopback-only, unauthenticated remote-code-execution surface by design.
- Streaming token output per task, so a long worker turn is visible rather than
  opaque.
- Git integration: no branch-per-task, no commit, no automatic revert of a bad
  cycle.
- A desktop wrapper, beyond the CLI and web UI that already exist.