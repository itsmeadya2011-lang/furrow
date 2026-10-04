# Furrow Agent Architecture

## Overview
Furrow is an autonomous development loop: plan tasks, execute them in parallel, run tests,
repeat, feeding each cycle's failure output into the next plan.

This repository contains **two independent implementations** of that idea:

1. **The Kilo agents** in `.kilo/`, driven by the `/furrow` slash command. These run inside
   the Kilo agent runtime and are what this file documents.
2. **The Python package** in `furrow/`, with the `furrow start` and `furrow web` commands.
   See `README.md`.

They are not wired to each other. The Python orchestrator does not call these agents, and
`/furrow` does not launch the Python loop. They share an idea and some vocabulary, nothing
more.

## Kilo components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (`mode: primary` in `.kilo/agent/furrow.md`)
- **Role:** Runs the loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallelizable tasks
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Report and repeat

The planner is asked for a small number of independent tasks (1-5). That is guidance in the
planner's instructions, not an enforced cap: nothing rejects a plan for being longer. What
is bounded is concurrency — how many workers run at once. The Python package exposes that
bound as `FURROW_MAX_PARALLEL_TASKS` (default 5); the Kilo agents have no equivalent knob,
so keep plans small by instruction.

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint and type checks, and returns a pass/fail JSON verdict. It does
  not fix anything; a failure comes back to the orchestrator as input to the next cycle's
  plan, and the fix is planned and executed like any other task.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

## Files
```
README.md              # the Python package: install, config, loop, limitations
pyproject.toml         # packaging, dependencies, pytest/ruff/mypy configuration
furrow/                # the Python package
  config.py            # Settings, Plan, TaskModel, TestResult, Provider
  llm.py               # provider transport, retries, error taxonomy
  core/orchestrator.py # the loop: scheduling, state, termination, events
  agents/              # planner, worker, tester, prompts, JSON repair
  tools/fs.py          # workspace-confined filesystem layer
  web/server.py        # web UI, health probe, event websocket
  cli/main.py          # `furrow start` and `furrow web`
tests/                 # pytest suite
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
Open by default: `bash: allow`, `edit: allow`, `read: allow`. There is no `ask` rule.

## Usage
- Run `/furrow <goal>` to start the Kilo loop.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

The Python orchestrator has no interactive input channel. Its equivalent mechanism is
automatic: test failures, task failures, blocked tasks and plan errors are collected and
passed to the next cycle's planner and worker calls as a `feedback` argument.

## What actually ships today
- The parallel plan/execute/test loop, in both implementations.
- A CLI (`furrow start`, `furrow web`) for non-TUI interaction.
- A web UI that streams the Python run's real event stream over a websocket, bound to
  loopback by default.
- Sandboxed, workspace-confined filesystem helpers in `furrow/tools/fs.py`.

## Future Work
- A file-editing tool loop for the Python worker. `WorkerAgent` currently has no tools and
  can only describe a change; `furrow/tools/fs.py` is the foundation but is not wired in.
- Persistent state for goal and task tracking across sessions. Task state currently lives
  in memory for the duration of one run.
- Cost and token accounting.