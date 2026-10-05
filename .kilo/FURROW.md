# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs a plan -> execute -> test development loop against a goal, in two forms: a set of Kilo agents driven by the `/furrow` command, and a standalone Python package under `furrow/`.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Report and repeat

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`, plus a `rationale`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task. Does not spawn further subagents.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Fixes failures. Returns pass/fail JSON with `passed`, `summary`, `failures`.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`, and asks the user for a goal when `$ARGUMENTS` is empty. Runs in the foreground (`subtask: false`).

## Python implementation

The `furrow/` package implements the same three roles as a standalone CLI and web service.

| Module | Role |
| --- | --- |
| `furrow/core/orchestrator.py` | The plan -> execute -> test loop: dependency-level scheduling, concurrency bound, termination conditions, and progress events. |
| `furrow/agents/planner.py` | Calls the planner model and parses the reply into a `Plan`. |
| `furrow/agents/worker.py` | Sends one task to the worker model and returns its summary. |
| `furrow/agents/tester.py` | Test command detection and execution; the exit code is authoritative over the model's verdict. |
| `furrow/agents/parsing.py` | Tolerant JSON extraction from raw LLM replies (markdown fences, surrounding prose, balanced brace spans). |
| `furrow/agents/prompts.py` | The planner, worker, and tester prompt texts, including the required JSON shapes. |
| `furrow/llm.py` | Provider clients (Anthropic, OpenAI, Ollama), bounded retries, and async file helpers (`read_file`, `write_file`, `list_files`). |
| `furrow/config.py` | The `Provider`, `TaskStatus`, `TaskModel`, `Plan`, `TestResult` and `Settings` models, loaded from `FURROW_*` env vars and `.env`. |
| `furrow/cli/main.py` | The `furrow` command group: `start` and `web`. |
| `furrow/web/server.py` | FastAPI app: single-page UI, `/ws` event stream, `/health`, `/stop`. |

### Loop termination

The Python orchestrator halts on the first of these conditions:

1. The planner returns a plan with no tasks.
2. Every task completed and the tests passed.
3. Consecutive failing cycles exceed `MAX_CONSECUTIVE_FAILURES` (3). This bounds an otherwise unlimited run.
4. `FURROW_MAX_CYCLES` is positive and has been reached. `0` or a negative value means unlimited.

A cycle that raises counts as a failed cycle rather than ending the run.

### Progress events (`on_event`)

`Orchestrator(goal, client=None, on_event=None)` accepts an optional `on_event` callback. The contract is a **synchronous** callable receiving one `dict` per event:

- every event has a `"type"` key and a `"cycle"` key (the cycle counter at emit time);
- the emitted types are `started`, `cycle`, `plan`, `task`, `tests`, `stopped`, and `error`;
- exceptions raised by the callback are swallowed, so a broken consumer can never break the run.

Because the callback is synchronous, the web server's integration buffers events on an unbounded queue and flushes them from a dedicated sender task, which keeps the orchestrator from blocking on a slow client and preserves event order.

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
furrow/                # Python package (see the module table above)
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop through the Kilo agents.
- The agent-driven loop halts on the first condition that fires: the planner returns no tasks; every task completed and the tests passed; more than 3 cycles failed back to back; or the cycle limit (`FURROW_MAX_CYCLES`) is positive and has been reached (`0` or less means unlimited). The user can also halt it at any time with Ctrl+C or by closing the session.
- Mid-loop user input is incorporated into the next cycle's plan.
- Run `furrow start "<goal>"` or `furrow web` to use the Python implementation instead. See the repository README for its options, `FURROW_*` settings, and the orchestrator's stop conditions.

## Future Work
- Desktop wrapper for non-browser, non-TUI interaction. No GUI toolkit is declared in `pyproject.toml` or imported anywhere in `furrow/`.
- Persistent state file for goal/task tracking across sessions. Orchestrator state (`cycles`, `plan`, `test_result`) lives only in memory for the lifetime of a run.
- Make the Python worker apply its edits to the workspace. Today `furrow/agents/worker.py` returns the model's summary as the task result and does not write files, even though `LLMClient.write_file` exists.
