# Furrow Architecture

This document describes both the **Kilo integration** (orchestrator/subagents/commands/config used in the Kilo environment) and the **Python `furrow/` package** (the standalone CLI/orchestrator/worker/tester/web stack). They are distinct layers that share the same name.

## Components

### Kilo integration (`.kilo/`)

The Kilo-side setup defines primary/subagents and a slash command for running Furrow inside Kilo:

| File | Type | Role |
| --- | --- | --- |
| `.kilo/agent/furrow.md` | Primary agent | Orchestrator for the infinite development loop. Coordinates planning, spawning workers, testing, and repeating. |
| `.kilo/agent/furrow-planner.md` | Subagent | Breaks a high-level goal into parallelizable tasks with IDs, descriptions, files, and dependencies. |
| `.kilo/agent/furrow-worker.md` | Subagent | Executes a single assigned task with targeted changes. |
| `.kilo/agent/furrow-tester.md` | Subagent | Runs tests/lint/type checks, fixes failures when possible, and returns pass/fail results. |
| `.kilo/command/furrow.md` | Slash command | Entry point (`/furrow <goal>`). Invokes the `furrow` orchestrator with `$ARGUMENTS`. |
| `kilo.json` | Project config | Sets the default agent to `furrow` and opens permissions for `bash`/`edit`/`read`. |

### Python package (`furrow/`)

| Module | Purpose |
| --- | --- |
| `furrow/config.py` | Pydantic models and settings. Enums: `Provider` (`anthropic`/`openai`/`ollama`), `TaskStatus`. Models: `TaskModel`, `Plan`, `TestResult`, `WorkerResult`, `RunResult`. `Settings` uses `FURROW_` prefix and reads `.env`. `Settings.state_path(workspace=None)` resolves `<workspace>/<state_file>`. |
| `furrow/llm.py` | Provider-neutral `LLMClient` with `complete()` and `chat(messages, tools, system, model)`. Converters for Anthropic/OpenAI message/tool formats. Supports Anthropic, OpenAI, and Ollama (OpenAI-compatible `/v1`). Tenacity retry on transient errors (408/409/425/429/5xx), non-retryable on 4xx. Helpers: `read_file`/`write_file`/`list_files`. |
| `furrow/agents/parsing.py` | Robust parsing: `strip_code_fences`, `extract_json`, `parse_json`, `parse_plan`, `parse_test_result`. Handles fences, prose around JSON, and nested structures. |
| `furrow/agents/planner.py` | `PlannerAgent.plan(goal, context="") -> Plan`. Retries once with stricter instructions on parse failure. |
| `furrow/agents/worker.py` | Core of the product. `WorkerAgent(task, client, settings, workspace, goal).run() -> WorkerResult`. Runs a real tool-calling loop with four tools: `read_file`, `write_file`, `list_files`, `run_command`. Paths are sandboxed to the workspace; `run_command` has a destructive-command denylist and a timeout; iteration budget is `max_worker_iterations`. |
| `furrow/agents/tester.py` | `TesterAgent.run(goal, tasks) -> TestResult`. `TesterAgent.detect_command() -> list[str] | None` detects per project type (pytest/npm/pnpm/yarn/cargo/go) instead of guessing binaries. Falls back to local classification on LLM failure. |
| `furrow/core/orchestrator.py` | `Orchestrator(goal, client, settings, state_store, event_bus, resume)`. Tracks tasks for real, executes in dependency waves honoring `dependencies`, bounds concurrency with `max_parallel_tasks`, honors `max_cycles` (0 = unlimited) with a no-progress guard, keeps `original_goal` intact and amends a separate working goal with test-failure feedback, persists state after every cycle, emits events, prints console progress. `run() -> RunResult`, `status() -> dict`. |
| `furrow/core/events.py` | `Event`, `EventType`, `EventBus` (fan-out pub/sub). |
| `furrow/core/state.py` | `RunState`, `HistoryEntry`, `StateStore`. Atomic writes (temp file + `os.replace`); corrupt state files are ignored. |
| `furrow/cli/main.py` | Click CLI: `furrow start`, `furrow status`, `furrow resume`, `furrow web`. |
| `furrow/web/server.py` | FastAPI app: `GET /` (single-page UI), `GET /api/state`, `GET /api/status`, `POST /api/start`, `WS /ws` streaming orchestrator events with final `{"type": "done", ...}` frame. |

## File tree

```text
furrow/
├── __init__.py
├── config.py
├── llm.py
├── agents/
│   ├── __init__.py
│   ├── parsing.py
│   ├── planner.py
│   ├── prompts.py
│   ├── tester.py
│   └── worker.py
├── cli/
│   ├── __init__.py
│   └── main.py
├── core/
│   ├── __init__.py
│   ├── events.py
│   ├── orchestrator.py
│   └── state.py
└── web/
    ├── __init__.py
    └── server.py
```

## Data flow (one cycle)

1. **Plan** — Orchestrator calls `PlannerAgent.plan(goal, context=feedback)`. Context includes test failures from the previous cycle if present.
2. **Merge** — `Orchestrator._merge_plan()` updates existing tasks (preserving completed status where appropriate) and sets the active task set.
3. **Execute** — `_plan_waves()` computes dependency waves. Tasks are executed concurrently per wave up to `max_parallel_tasks`. For each task, `WorkerAgent.run()` runs a tool-calling loop (`read_file`/`write_file`/`list_files`/`run_command`) until no tool calls or iteration budget exhausted. Results update `TaskModel.status/result/error/attempts/timestamps`.
4. **Test** — `TesterAgent.run(original_goal, tasks)` detects the test command (or falls back to classification) and returns `TestResult`.
5. **Persist & emit** — Events are emitted throughout (cycle/task/test/state). `RunState` (goal/original_goal/cycles/tasks/history) is saved atomically to `state_path()`. If tests fail, `goal` is amended with failure details for the next cycle; a no-progress guard can stop the run.

## Events and state

- **Events** (`furrow/core/events.py`): `EventType` values include `CYCLE_STARTED`, `PLAN_CREATED`, `TASK_STARTED`, `TASK_FINISHED`, `TEST_STARTED`, `TEST_FINISHED`, `STATE_SAVED`, `LOG`, `ERROR`. `EventBus` provides fan-out pub/sub. The web UI and CLI consume these events.
- **State file**: `<workspace>/.furrow/state.json` (configurable via `FURROW_STATE_FILE`). Contains `RunState` with `goal`, `original_goal`, `cycles`, `tasks`, `history`, and `updated_at`. Written atomically; corrupt files are ignored on load.

## Permissions

Kilo integration uses open permissions by default (`bash: allow`, `edit: allow`, `read: allow`) as configured in `kilo.json`.

## Version control

A Furrow run works in a disposable workspace, so a change is only durable once it is committed **and** pushed to `origin`. A local-only commit is lost when the workspace is discarded.

- Work on the branch the session was given (`kilo/<name>`). Do not rewrite history on a shared ref — no force pushes, no rebases of published branches.
- Commit at the end of every cycle that produced changes. The message describes the change, not the process.
- Push the branch to `origin` before the session ends, and set it to track the remote branch.
- If the session was given a base branch (a `session/agent_<uuid>` branch), fast-forward that branch to the run's tip and push it too, so the next run starts from the improved code rather than re-deriving it. A fast-forward is safe because the old tip stays in history.
- Leave `.furrow/` (run state) untracked — it is local scratch state, covered by `.gitignore`.

## Usage

- **Kilo**: Run `/furrow <goal>` to start the Kilo orchestrator loop.
- **CLI**: `furrow start "<goal>"`, `furrow status`, `furrow resume`, `furrow web` (default http://127.0.0.1:8000).
- **Web API/WebSocket**: `GET /api/state`, `GET /api/status`, `POST /api/start`, `WS /ws` streams events and ends with `{"type": "done", ...}`.

## Future Work

- Richer web UI with per-agent panels
- A richer `furrow` package plugin system
- Multi-provider model routing
- Better cost accounting
