# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs a development loop for web and
CLI projects. A planner agent decomposes the goal, worker agents execute the
tasks with real tools against a sandboxed workspace, and a tester agent runs the
project's own checks. The loop is bounded, not open-ended: it halts on
completion, on a cycle cap, on stall detection, or when stopped.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Report and repeat
- **Concurrency:** tasks are grouped into dependency waves. Tasks whose
  `dependencies` are satisfied run together under a semaphore bounded by
  `max_parallel_tasks`; a later wave waits for the tasks it depends on.
- **Halt conditions** (first one met wins, reported as `halted_reason`):
  - `stopped` - the user stopped the session.
  - `max_cycles` - `max_cycles` is greater than zero and was reached.
  - `stalled` - `max_stalled_cycles` consecutive cycles made no progress
    (no new task, no completed task, no test verdict change).
  - `goal_complete` - the planner returned an empty task list.
- **Focus:** the original goal is never overwritten. Test failures accumulate
  into a separate, bounded focus string that is passed to the next plan, so the
  goal stays intact while the failures stay visible.
- **Isolation:** a planner or tester exception becomes an `ERROR` event and a
  stalled cycle. It never ends the run.

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`,
  `dependencies`, plus a `rationale`.
- **Dependencies are honoured:** the orchestrator schedules tasks in waves, so
  a task listed in another task's `dependencies` never starts first.
- An empty task list is the signal that the goal is already met. A reply that
  is not a usable plan is a failure, never a silently empty one.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task with real tools. Minimal, targeted
  changes. No scope creep.
- **Tools:** `read_file`, `write_file` (atomic temp file plus rename),
  `edit_file` (exact-string replace, refusing a missing or ambiguous match
  unless `replace_all`), `list_files`, `search`, `run_command`.
- **Sandbox:** every path is resolved against the workspace root and refused
  when it escapes - `../` traversal, an absolute path outside the root, or a
  symlink pointing out.
- **Bound:** the loop is capped by `max_tool_iterations`. Reaching the cap
  returns an explicit `INCOMPLETE` summary instead of a false success.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs the project's tests, linter and type checker and returns a
  pass/fail verdict with a summary and a failure list. It reports failures; it
  does not fix them. The orchestrator feeds the failures into the next cycle's
  focus and the next set of workers repairs the code.
- **Ground truth:** each command is auto-detected from an executable on PATH
  plus a project marker file, or taken verbatim from `test_command` /
  `lint_command` / `typecheck_command`. Commands run inside the workspace with
  no shell and a timeout; a timeout is a failure, not a pass. A verdict that
  contradicts a non-zero exit code is overridden to failed. If no test command
  can be resolved, verification is reported as skipped.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from
  `$ARGUMENTS`.

## Files
```
.kilo/
  agent/
    furrow.md          # Orchestrator
    furrow-planner.md  # Planner subagent
    furrow-worker.md   # Worker subagent
    furrow-tester.md   # Tester subagent
  command/
    furrow.md          # /furrow slash command
kilo.json              # Project config (default agent = furrow, open permissions)
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The loop halts when the goal is complete, when the cycle cap is hit, when it
  stalls, or when the user stops the session.
- There is no mid-loop input channel: a new instruction takes effect on the
  next run, not inside a running one.

## Python Implementation
The Kilo agents above and the Python package are separate things. The package
is the product; it ships the same loop with its own agents.

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

- **Live activity:** every cycle, task, tool call, test result and error is
  published on an `EventBus` with a bounded history ring buffer. Payloads are
  small and JSON-serialisable. A `tool_call` message is assembled from the
  allowlist `PREVIEW_SAFE_ARGS` in `agents/tools.py` and capped at
  `PREVIEW_CHARS` (240) for the whole message, tool name included, so the content
  arguments - `write_file.content`, `edit_file.old_string`/`new_string` - and
  `search.query` (a literal substring search's query is text copied out of a file
  by construction) never reach the message, the `data` payload or
  `Event.to_dict()`, on success or on error. `run_command.command` is a
  deliberate exception: it renders head/tail-clipped in the message and is echoed
  in `data["command"]` as a pre-existing tool detail field, because dropping it
  from the message would not close the route. Consequence: never put a secret in a
  shell command, and note that `path`/`pattern` render verbatim. The CLI renders
  the feed as the run proceeds.
- **State:** with `state_file` configured, a JSON snapshot is written atomically
  after every cycle. Resuming a run from that file is not implemented.
- **Web UI:** `furrow web` serves a single-page monitoring UI plus a REST API
  and a websocket event stream. It binds `127.0.0.1` by default and expects
  `FURROW_WEB_TOKEN` to be set before it is exposed anywhere else. A blank or
  whitespace-only `FURROW_WEB_TOKEN` is treated as unset - auth off, docs open -
  and logs an `ERROR`-level startup warning so that is never silent. The
  interactive docs (`/docs`, `/redoc`, `/openapi.json`) are served only while
  that token is unset; once it is set they answer 401 without a credential and
  404 with one.

## Future Work
- Desktop wrapper around the Python CLI and web surface.
- Resuming a run from the persisted state file.