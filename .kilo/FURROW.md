# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent. The Kilo markdown agents described here run a development loop that only the user stops. The Python package described below runs the same plan/execute/test shape but has explicit termination rules and bounded parallelism.

This document covers the Kilo markdown-agent side of Furrow. The same architecture also exists as a runnable Python package under `furrow/`; see "Python Implementation" below.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the loop. Receives goals, plans, delegates, tests, and repeats.
- **Step budget:** 50
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Report and repeat
- **Concurrency:** the orchestrator spawns every planned worker in parallel and waits for all of them. There is no wave scheduling and no cap beyond the planner's own 1-5 task limit; `dependencies` are carried in the plan JSON but the markdown loop does not order execution by them.
- **Termination:** none in the agent itself. `.kilo/agent/furrow.md` states the loop runs forever and stops only when no tasks remain or the user interrupts.

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Step budget:** 10
- **Output:** JSON with `tasks[]` (each having `id`, `description`, `files`, `dependencies`) and `rationale`. There is no `complete` flag in this contract.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Step budget:** 15
- **Constraint:** 1-3 tool call rounds per task.
- **Editing:** the markdown worker edits the repository directly with Kilo's own tools.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Fixes failures. Returns pass/fail JSON.
- **Step budget:** 15
- **Output:** JSON with `passed`, `summary`, `failures`.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`, asking the user for one if the argument is empty. Runs as a top-level task (`subtask: false`).

## Python Implementation

The five roles above are mirrored by a Python package in the same repository: `Orchestrator` (`furrow/core/orchestrator.py`), `PlannerAgent` (`furrow/agents/planner.py`), `WorkerAgent` (`furrow/agents/worker.py`), and `TesterAgent` (`furrow/agents/tester.py`), with their prompts in `furrow/agents/prompts.py`.

Shared with the markdown side: both plan 1-5 independent tasks per cycle, cap each task at 1-3 tool call rounds, and exchange the same JSON contracts (`tasks[]`/`rationale` for plans, `passed`/`summary`/`failures` for test results).

The differences are intentional, and they are the load-bearing part of the Python design:

- **`Orchestrator.__init__(goal, client=None, on_event=None)`.** `goal` is immutable for the whole run; what changes each cycle is `current_focus`, a rebuilt string containing the original goal, a bounded list of already-completed task summaries, and the latest test failures. The objective cannot drift onto whatever failed last.
- **Dependency-ordered waves.** `Orchestrator._execute_tasks` runs the plan in waves. A task is runnable only when every dependency it names is either already completed or absent from the plan. A task whose dependency failed is set to `TaskStatus.BLOCKED` and is not attempted; a plan whose dependencies form a cycle is detected (no task becomes runnable) and every remaining task is blocked rather than hanging.
- **Concurrency cap.** Within a wave, `_run_wave` holds an `asyncio.Semaphore` sized by `settings.max_parallel_tasks`, so a plan of five tasks with a cap of two never has more than two workers in flight.
- **Explicit termination rules.** `_is_done()` halts the run when `max_cycles` is reached (`0` means unlimited), when the planner returns an empty task list, when the planner sets `complete: true` and the suite is not known to be red, or when `max_idle_cycles` consecutive cycles made no progress (`0` means unlimited). `run()` re-checks the cycle cap afterwards as a backstop, then prints a summary panel.
- **`on_event` hook.** Every message the loop emits is printed and also passed to `on_event`, with rich renderables flattened to plain text first. A consumer that raises is logged and ignored. `furrow/web/server.py` is the only consumer today: it strips rich markup and streams the text to the browser over `/ws`, which is what makes the web monitor possible. The CLI passes no hook.
- **The `complete` flag.** The Python `Plan` model carries `complete`, which the markdown plan contract does not, and that flag is what lets the Python loop terminate cleanly.
- **Workers do not edit files.** `WorkerAgent` returns a text summary and never writes; the summary is stored as the task result and is the only part of its output that survives into the next cycle's planning context. `LLMClient.write_file` and `LLMClient.list_files` exist but nothing in the loop calls them. The markdown workers do edit the repository.
- **The exit code is authoritative.** `TesterAgent` runs a subprocess and lets the model summarise the output, but `TestResult.passed` is always derived from the return code; the model can never flip the verdict.

## Files

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml            # ruff check, ruff format --check, pytest (blocking), mypy (continue-on-error)
├── .kilo/
│   ├── .gitignore            # Kilo-local ignores (node_modules, lockfiles, agent-manager.json)
│   ├── FURROW.md             # This document
│   ├── agent/
│   │   ├── furrow.md         # Orchestrator (primary, steps: 50)
│   │   ├── furrow-planner.md # Planner subagent (steps: 10)
│   │   ├── furrow-worker.md  # Worker subagent (steps: 15)
│   │   └── furrow-tester.md  # Tester subagent (steps: 15)
│   └── command/
│       └── furrow.md         # /furrow slash command
├── furrow/                   # Python package
│   ├── __init__.py
│   ├── config.py             # Settings + Plan/TaskModel/TaskStatus/TestResult models
│   ├── llm.py                # Async LLM client (anthropic/openai/ollama) and file helpers
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner.py
│   │   ├── prompts.py
│   │   ├── tester.py
│   │   └── worker.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py           # `furrow` console script entry point (start, web, --version)
│   ├── core/
│   │   ├── __init__.py
│   │   └── orchestrator.py   # The development loop
│   └── web/
│       ├── __init__.py
│       └── server.py         # FastAPI app behind `furrow web` (/, /health, /ws)
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # FakeLLM, env isolation, sys.path guard
│   ├── test_agents.py
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_llm.py
│   ├── test_orchestrator.py
│   └── test_web.py
├── .gitignore
├── kilo.json                 # Project config (default agent = furrow, open permissions)
├── pyproject.toml            # Packaging, dependencies, pytest and ruff config
└── README.md                 # Python package docs (install, config, architecture)
```

## Runtime State

Furrow does not persist any runtime state today. Plans and task results live only in the orchestrator's memory for the life of the process, and nothing in the repository or the Python package reads or writes a state file. The Python `Orchestrator` holds `plan`, `tasks`, `completed_summaries` and the last `TestResult` as instance attributes and drops them when the process exits; the loop is bounded by `max_cycles` and `max_idle_cycles` rather than by being able to resume. The only file-writing code in the package is the generic `LLMClient.write_file` helper, which no agent calls. Plan and task state that survives a restart is listed under "Future Work".

The name `furrow` is currently overloaded: it is the distribution name in `pyproject.toml`, the import package name (`furrow/`), and the console script name (`furrow = "furrow.cli.main:main"`). A future runtime state directory would want to live at `furrow/` at the repository root, which is where the Python package already sits. Keeping the two distinct — for example by moving state under `.furrow/` or `var/furrow/` — is a candidate for renaming once state exists.

The distribution name also collides on PyPI: `furrow` there is an unrelated text-formatting package, so the Python side is installed from a checkout rather than from an index.

## Permissions

Open by default: `bash: allow`, `edit: allow`, `read: allow`, configured in `kilo.json`. Note that the orchestrator prompt in `.kilo/agent/furrow.md` refers to an `allow`/`ask` fallback; no `ask` fallback is configured for this project, all three permissions are set to `allow`. This is a documentation discrepancy in the prompt, not a configuration gap, and it is left in place rather than edited here.

## Usage
- Run `/furrow <goal>` to start the loop.
- The markdown agent runs until the goal is complete or the user stops the session with Ctrl+C.
- Mid-loop user input is incorporated into the next cycle's plan.
- Outside Kilo, `furrow start "<goal>"` and `furrow web` drive the Python implementation of the same loop; see `README.md`. The Python loop stops on its own once a termination rule fires, and Ctrl+C exits cleanly with status 0.

## Future Work
- Persistent goal and task state that survives a process restart.
- Real file-editing tool execution for Python workers; today `WorkerAgent` only returns a text summary and does not write files itself.
- A desktop wrapper for non-terminal interaction.
- Richer run history and telemetry (per-cycle timings, token usage, failure classification).
