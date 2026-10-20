# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects.

There are two halves to Furrow:

- the **Kilo agents** below (`.kilo/`), which drive the loop inside a Kilo session
- the **`furrow` Python package** (`furrow/`), the standalone installable implementation

Both implement the same plan -> execute -> test -> repeat cycle.

## Kilo Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the infinite loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Report and repeat

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
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## Python Package (`furrow/`)

The installable implementation. `pip install furrow`, then `furrow start "<goal>"`.

### Orchestrator (`furrow/core/orchestrator.py`)
Drives the loop: plan -> execute -> test -> repeat.

- Tracks the current `Plan` on the instance, so completion is detected per plan
  rather than assumed.
- Keeps the original `goal` immutable; the per-cycle remediation text lives in
  a separate `focus` attribute.
- Executes tasks with a dependency-aware scheduler bounded by
  `settings.max_parallel_tasks`; dependency cycles are detected and run rather
  than deadlocking.
- Retries the planner three times with backoff, and stops gracefully when the
  planner cannot produce a plan.
- Stops on `settings.max_cycles`, and after three consecutive no-progress
  cycles (a stall guard that prevents an infinite hot loop).
- Emits lifecycle events through an optional async `on_event` callback. The
  web UI consumes this to stream live progress.

### Agents (`furrow/agents/`)
- **Planner** - parses the model response with a fence/prose-tolerant JSON
  extractor, retries once with a stricter instruction, and validates the
  result against the `Plan` model.
- **Worker** - runs a bounded tool loop (8 rounds) with `read_file`,
  `write_file` and `list_files`. Every path is resolved against the workspace
  root and anything that escapes it is refused, so workers cannot modify files
  outside the workspace.
- **Tester** - detects the test runner with `shutil.which`, runs it with `cwd`
  set to the workspace, and derives pass/fail from the **exit code**. The model
  only summarises failures and can never turn a failing suite into a passing
  one.

### Provider layer (`furrow/llm.py`)
- Supports Anthropic, OpenAI and Ollama through one interface.
- Retries transient failures (connection errors, timeouts, rate limits, 5xx)
  with exponential backoff; client errors are not retried.
- Exposes `complete`, `complete_with_tools` and `run_tool_loop` so agents can
  use tools without knowing which provider is in play.
- File helpers use explicit UTF-8 and skip VCS, cache and build noise when
  listing directories.

### Supporting modules
- `furrow/config.py` - `Settings`, `Plan`, `TaskModel`, `TestResult`. All
  settings are overridable via `FURROW_`-prefixed environment variables.
- `furrow/logging.py` - structlog configuration driven by `FURROW_LOG_LEVEL`.
- `furrow/cli/main.py` - the `furrow start` and `furrow web` commands.
- `furrow/web/server.py` - FastAPI dashboard that streams orchestrator events
  to the browser over a websocket.

## Quality Gates

```bash
pytest -q                 # test suite
ruff check .              # lint
ruff format --check .     # formatting
mypy furrow               # type check
```

All four are clean. The suite covers the config models, the provider layer, the
planner, the worker tools and their sandboxing, the tester, the orchestrator
loop, the CLI and the websocket API. No test touches the network or a live
model.

## Future Work
- Persistent state file for goal/task tracking across sessions.
- CLI wrappers for non-TUI interaction beyond `furrow start` / `furrow web`.
- Desktop wrapper for monitoring parallel agent activity.
- Streaming token-by-token output in the web UI.
