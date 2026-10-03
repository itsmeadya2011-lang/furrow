# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an unbounded development loop for web/CLI/desktop projects. It exists as two independent layers:

- **Kilo agent layer** — the definitions in `.kilo/` plus `kilo.json` at the repository root. The orchestrator and its planner/worker/tester are Kilo agents; the orchestrator spawns workers as subagents with the Task tool, so they run real tools and edit files in the session.
- **Python package** — the `furrow/` code in this repository. The same loop as a standalone, `pip`-installed program (`furrow start`, `furrow web`) built on in-process coroutines, with no Kilo involvement.

Everything below describes the Kilo layer unless it explicitly says "Python package".

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (`kilo.json` sets `default_agent`); also usable as a subagent
- **Role:** Runs the loop. Receives a goal, plans, delegates, verifies, repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (at most 5; see Scheduling)
  3. Execute via parallel `furrow-worker` subagents (Task tool)
  4. Verify via the `furrow-tester` subagent, or by running the project's own checks directly
  5. Report and repeat

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`, plus a `rationale`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs the test suite, linter and type checker, diagnoses and **fixes** failures, returns pass/fail JSON (`passed`, `summary`, `failures`).
- **Not the Python tester:** `furrow/agents/tester.py`'s `TesterAgent` shares the name and the JSON shape but not the job. It runs the project's detected test commands plus `ruff check .` and `mypy .` when they are installed, takes exit codes as authoritative, returns a vacuous pass when no runner is detected, and **fixes nothing** — the orchestrator feeds its `failures` into the next cycle's focus. Never describe its verdict as if it had repaired anything.

## Scheduling
- **Tasks per plan:** at most 5. In the Python package this is a hard cap; the plan is clamped to `max(1, min(5, max_parallel_tasks))` tasks, so `max_parallel_tasks` can lower it but never raise it.
- **Dependencies:** `dependencies` lists ids from the same plan's `tasks` array. The Python runner honours them: a task starts only once every task it depends on has completed, and the task is handed those dependencies' results.
- **Waves:** all runnable tasks of a plan go out together, one in-process `WorkerAgent` coroutine each (`asyncio.gather`), bounded by an `asyncio.Semaphore` of `max_parallel_tasks` tasks at a time. `max_parallel_tasks = 0` means no cap.
- **Unrunnable work:** tasks caught in a dependency cycle are skipped before execution, as is any task whose dependency failed or was skipped. The Python planner additionally drops unknown and self-referential ids and breaks cycles, so a cycle should never reach the runner.
- **Loop budget:** `max_cycles = 0` (the default) means the loop is unbounded; any positive value is a hard budget. Completion requires the most recent verification to have passed **and** every task in the final plan to be completed.

## Files
```
kilo.json                 # Project config (default agent = furrow; bash/edit/read = allow)
README.md                 # Front page of the Python package (`furrow` CLI and web UI)
pyproject.toml            # Package metadata; entry point furrow = "furrow.cli.main:main"
furrow/
  core/orchestrator.py    # The loop
  agents/planner.py       # Goal -> plan
  agents/worker.py        # One task, grounded in real file contents
  agents/tester.py        # Runs the project's tooling, returns a verdict
  agents/prompts.py       # The three agent prompts
  config.py               # Settings (FURROW_* env vars, .env)
  llm.py                  # anthropic / openai / ollama chat client
  cli/main.py             # `furrow start`, `furrow web`
  web/server.py           # Web UI + websocket
tests/
.kilo/
  FURROW.md               # This file
  agent/
    furrow.md             # Orchestrator agent
    furrow-planner.md     # Planner subagent
    furrow-worker.md      # Worker subagent
    furrow-tester.md      # Tester subagent
  command/
    furrow.md             # /furrow slash command
```

## Permissions
`kilo.json` grants exactly `bash: allow`, `edit: allow`, `read: allow`. It contains no `ask` entries and no entries for any other permission, so anything unlisted falls back to Kilo's own default behavior — there is no `ask` fallback configured in this repository.

## Usage
- Run `/furrow <goal>` to start the loop in a Kilo session; with no goal, the command asks for one first.
- The loop runs until the goal is complete or the user stops the session.
- **Input, Kilo layer:** the user can interject mid-session and the orchestrator is instructed to fold that into the next plan.
- **Input, Python package:** there is no mid-loop input channel. `furrow start` prompts only for the goal, before the first cycle, and the web UI offers Start and Stop — it cannot change the goal or hand the loop new instructions.

## Future Work
The original roadmap listed a non-TUI CLI and a web UI for monitoring parallel activity. Both exist now (`furrow/cli/main.py`, `furrow/web/server.py`, wired to the orchestrator's event stream). Still absent from the source:
- **Persistent state.** Goal, focus and task board live in one orchestrator instance; nothing survives a restart.
- **A tool-calling worker in the Python package.** `WorkerAgent` is not tool-calling — it sends one grounded prompt and receives a single JSON envelope — but it does apply file edits: validated `files` entries are written to the workspace with path checks, backups to `<backup_dir>/<UTC timestamp>-<task id>/<relative path>`, size caps (`max_files_per_task`, `max_write_bytes`, `max_total_write_bytes`) and advisory mode (`Settings.apply_changes` / `FURROW_APPLY_CHANGES=0`). The code-editing worker remains the Kilo subagent.
- **Web UI authentication.** The server is unauthenticated and binds loopback by default.