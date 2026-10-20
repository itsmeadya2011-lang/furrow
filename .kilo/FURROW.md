# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs a plan -> execute -> test loop over a software project. A planner breaks the goal into parallelizable tasks, workers implement them, and a tester verifies the result. The loop then repeats with what was learned, until there is nothing left to do — it is not unconditional.

The project has two halves that share one design:
- The agent architecture in `.kilo/` — the orchestrator, planner, worker and tester agents, driven by the `/furrow` slash command inside a Kilo session.
- The Python package in `furrow/` — the same loop as a standalone program, with a non-TUI CLI, a web UI and a persisted run state.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Report and repeat, for as long as the goal is not yet met
- **Stopping:** The loop ends when the goal is complete, when there is nothing left to plan, or when the user stops the session. The Python orchestrator additionally enforces a cycle budget (`max_cycles`) and gives up when the loop stops making progress (`max_consecutive_failures`).

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
- **Role:** Runs tests, lint, type checks. Fixes failures. Returns pass/fail JSON.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

## Python package

The package implements the same loop outside a Kilo session, so a run can be driven headless, inspected later, or monitored in a browser.

| Agent architecture | Python package |
| --- | --- |
| `furrow` orchestrator | `furrow/core/orchestrator.py` |
| `furrow-planner` | `furrow/agents/planner.py` |
| `furrow-worker` | `furrow/agents/worker.py` |
| `furrow-tester` | `furrow/agents/tester.py` |

The difference is who does the work: the Kilo agents delegate to subagents through the Task tool and edit files with their own tools, while the package calls the LLM API itself and writes the files.

- **`furrow/core`** — `orchestrator.py` drives the loop and keeps real state across cycles (`tasks` holds every task ever planned, deduped by id; `history` holds one entry per cycle; `cycles` counts them). `events.py` defines the progress events it emits: `cycle_start`, `plan`, `task_start`, `task_done`, `task_failed`, `test_result`, `cycle_end`, `done`. `state.py` serialises the run to a JSON state file and restores it.
- **`furrow/agents`** — the planner, worker and tester. The worker really edits files: it reads the task's files plus a workspace listing, asks the model for `{summary, edits: [{path, content}]}` and writes the files inside the workspace, rejecting absolute paths and `../` escapes. `apply_worker_edits` is the kill switch. The tester picks the runner that matches the project (pytest, npm, cargo, go) instead of the first one that exists. `parsing.py` extracts JSON from responses that arrive wrapped in markdown fences or padded with prose.
- **`furrow/llm`** — provider client for Anthropic, OpenAI and Ollama, with retries and backoff on transient errors.
- **`furrow/cli`** — the `furrow` console script: `start`, `status`, `resume`, `web`. This is the non-TUI wrapper.
- **`furrow/web`** — a FastAPI app that streams the orchestrator's events over a websocket and exposes a REST API (`GET /api/health`, `GET /api/state`, `POST /api/start`, `POST /api/stop`). The page renders a live event log and task list for the parallel workers; the run continues server-side when the browser disconnects.
- **Run state** — `furrow start` and `furrow resume` write the goal, tasks and per-cycle history to the state file (default `.furrow/state.json`) after every cycle. `furrow status` reads it back and `furrow resume` continues the run from it, which is the goal/task tracking across sessions that the agent side keeps in its context.

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
  FURROW.md            # This document
kilo.json              # Project config (default agent = furrow, open permissions)
furrow/
  cli/main.py          # Console script: start / status / resume / web
  core/orchestrator.py # Plan -> execute -> test loop
  core/events.py       # Progress event stream
  core/state.py        # Run state: save / load / restore
  agents/planner.py    # Planner
  agents/worker.py     # Worker (applies file edits)
  agents/tester.py     # Tester (runs the project's test runner)
  agents/prompts.py    # Prompts for the three agents
  agents/parsing.py    # Tolerant JSON extraction
  llm.py               # Anthropic / OpenAI / Ollama client
  web/server.py        # FastAPI app: page, websocket, REST API
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The loop ends when the goal is complete, when there is nothing left to plan, or when the user stops the session (Ctrl+C or closing the session).
- Mid-loop user input is incorporated into the next cycle's plan.
- The same loop outside a Kilo session is the Python package:
  - `furrow start "<goal>"` — run the loop, saving state after every cycle.
  - `furrow status` — show the saved run.
  - `furrow resume` — continue the saved run.
  - `furrow web` — live UI and REST API on http://localhost:8000.
  - `max_cycles` bounds the number of cycles and `max_consecutive_failures` stops a loop that completes no task, so the package never spins forever.

## Future Work
- Desktop wrapper — the CLI and the web UI exist, but there is no native desktop client.
- Dependency-aware scheduling — `dependencies` are planned and persisted, but the executor runs tasks in plan order without waiting on them.
- Persisting web UI runs — `furrow start` and `furrow resume` write the state file after every cycle; runs started from the web UI are not written to it.
- Patch-based edits and review — workers rewrite whole files, and there is no diff review step before edits are applied.
