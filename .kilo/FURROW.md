# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs a development loop: plan a goal
into independent tasks, execute them in parallel, test the result, and repeat
until the goal is satisfied or the user stops it.

There are two implementations of this design. The Python package in `furrow/`
is the distributable one (`pip install .`, entry point `furrow`). The agent
definitions in `.kilo/agent/` are a separate, TUI-side implementation of the
same four roles — richer in places, but not interchangeable with the Python
port, and they are not kept in sync.

## Components

### 1. Orchestrator (`furrow/core/orchestrator.py`)
- **Type:** Primary loop
- **Role:** Owns the cycle and the stop condition. Never edits files itself.
- **Loop:**
  1. Plan the goal (plus the current focus, if any) into tasks
  2. Fan the tasks out to workers, honoring declared dependencies and keeping at
     most `max_parallel_tasks` in flight
  3. Collect per-task results; a failed task is recorded, not raised
  4. Hand the plan and results to the tester
  5. On failure, make the failures the focus for the next cycle
  6. Repeat until the loop stops
- **Stop reasons** (`LoopResult.stopped_reason`): `goal-complete` (tests passed
  with no failed task), `no-tasks` (empty plan), `max-cycles` (`max_cycles`
  reached, `0` = unlimited), `stalled`, or `cancelled`. Only `goal-complete`
  counts as success.
- **Stall detection** (`STALL_LIMIT = 3`): three consecutive cycles in the same
  stuck state — the same test failures and failing tasks (compared as a
  whitespace-collapsed fingerprint, so cosmetic drift does not hide a stall),
  three consecutive planner failures, or three consecutive crashed cycles. It is
  a deliberate guard against looping forever and burning API credits; the
  repeated failures are still reported in `LoopResult.failures`.
- **Events:** every cycle boundary emits an event to an optional handler, which
  is how a UI follows the loop. Names: `run.start`, `cycle.start`, `plan`,
  `task.start`, `task.done`, `task.failed`, `task.skipped`, `test.result`,
  `cycle.done`, `run.done`.

### 2. Planner (`furrow/agents/planner.py`)
- **Type:** Agent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** A `Plan` of `TaskModel` entries (`id`, `description`, `files`,
  `dependencies`) plus a `rationale`.
- **Notes:** JSON is extracted leniently from the reply (bare, fenced, or
  embedded in prose) and normalized; unparsable replies are retried up to
  `MAX_PLAN_ATTEMPTS` with the parse error fed back.

### 3. Worker (`furrow/agents/worker.py`)
- **Type:** Agent
- **Role:** Executes exactly one assigned task. Minimal, targeted changes, no
  scope creep, no further delegation.
- **Input:** The task's current file contents (budgeted) plus a bounded listing
  of the workspace.
- **Output:** `<<<FILE: path>>>` blocks to create or rewrite a file and
  `<<<EDIT: path>>>` blocks with `<<<SEARCH>>>`/`<<<REPLACE>>>` halves for
  targeted changes.
- **Constraint:** Model-supplied paths are resolved and refused unless they land
  inside the workspace, so the worker cannot write outside it.

### 4. Tester (`furrow/agents/tester.py`)
- **Type:** Agent
- **Role:** Runs the project's own test command in the workspace (bounded by
  `test_timeout`) and returns a `TestResult` (`passed`, `summary`, `failures`).
- **Verdict:** the subprocess exit code, not the output. The runner is chosen
  from the manifests present (pytest / npm / pnpm / yarn / cargo / go); if none
  is detected the cycle fails without asking the LLM. The LLM only re-phrases
  the summary and failures and can never flip `passed`.

### 5. Shared infrastructure
- `furrow/config.py` — `Settings` (env prefix `FURROW_`, reads `.env`) plus the
  `Plan` / `TaskModel` / `TestResult` models.
- `furrow/llm.py` — `LLMClient`: Anthropic and OpenAI (and Ollama via its
  OpenAI-compatible endpoint) with tenacity retries on transient failures, plus
  the async file read/write/list helpers the workers use.

### 6. Web UI (`furrow/web/server.py`)
- `GET /` serves a single page: a goal field and a live log.
- `GET /ws` streams one run. The client sends `{"goal", "model"?}` once; the
  server forwards every orchestrator event as one JSON message
  (`{"type", "cycle", "message", ...}`) and ends with `run.done`, or with
  `error` if the request was rejected before a run started.
- `POST /api/start` validates a goal and echoes the effective settings; it never
  starts a run.
- A client disconnect cancels the run and its in-flight workers.

## Entry points
- `furrow start "<goal>"` — the loop on the terminal. Flags: `--provider`,
  `--model`, `--planner-model`, `--worker-model`, `--tester-model`,
  `--max-cycles`, `--parallel`, `--workspace`, `--log-level`, `--debug`. Exits
  `0` only on `goal-complete`, `130` on `Ctrl-C`.
- `furrow web` — the same loop behind a FastAPI/WebSocket UI, bound to
  `127.0.0.1:8000` by default (`--host`, `--port`, `--log-level`). It has no
  authentication, so a non-loopback `--host` publishes an agent that runs
  commands and writes files.
- `/furrow <goal>` — slash command that starts the TUI-side orchestrator
  (`.kilo/agent/furrow.md`).

## Files
```
furrow/
  config.py            # Settings + data models
  llm.py               # provider clients, retries, file helpers
  core/orchestrator.py # the loop
  agents/              # planner, worker, tester, prompts
  cli/main.py          # click CLI: start / web
  web/server.py        # FastAPI app + websocket
.kilo/
  agent/
    furrow.md          # Orchestrator (TUI-side)
    furrow-planner.md  # Planner subagent
    furrow-worker.md   # Worker subagent
    furrow-tester.md   # Tester subagent
  command/
    furrow.md          # /furrow slash command
kilo.json              # Project config (default agent = furrow, open permissions)
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`. The Python
workers instead bound themselves by refusing writes outside `workspace`.

## Usage
- `furrow start "<goal>"` or `/furrow <goal>` to start the loop.
- The CLI runs until one of the stop reasons above; `Ctrl-C` ends the run
  cleanly (exit `130`). The TUI orchestrator loops until the user stops it.
- Mid-loop user input becomes the goal for the next cycle (TUI side).

## Future Work
- Persistent state file for goal/task tracking across sessions — not
  implemented; cycle state lives in memory for the life of the process.
- Run history or restart-from-cycle in the web UI — the run lives in memory and
  ends with the page, which cancels it on disconnect.
- Keeping the Python port and `.kilo/agent/` definitions from drifting.