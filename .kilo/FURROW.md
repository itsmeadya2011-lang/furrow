# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs a repeated development loop: it plans a goal into tasks, executes them in parallel through LLM worker agents, runs the project's own test suite, and feeds failures back into the next round of planning.

There are two implementations of this same conceptual design, and they share no code:
- The Python package under `furrow/`, which is the pip-installable product.
- The Kilo TUI-agent definitions under `.kilo/agent/`, plus the `/furrow` slash command in `.kilo/command/furrow.md`, configured by `kilo.json` at the repo root rather than inside `.kilo/`.

Because they are maintained separately they can drift, and keeping them in sync is a real maintenance obligation. Sections below describe the Python package unless stated otherwise; the Kilo-side differences are called out under Components and Usage.

## The Two Implementations
| | Python package | Kilo agents |
|---|---|---|
| Entry point | `furrow start` or `furrow web` | `/furrow <goal>` or the `furrow` agent |
| Orchestrator | `furrow/core/orchestrator.py` | `.kilo/agent/furrow.md` |
| Planner | `furrow/agents/planner.py` | `.kilo/agent/furrow-planner.md` |
| Worker | `furrow/agents/worker.py` | `.kilo/agent/furrow-worker.md` |
| Tester | `furrow/agents/tester.py` | `.kilo/agent/furrow-tester.md` |
| Concurrency limit | `FURROW_MAX_PARALLEL_TASKS` | none |

The Python side is a normal library with its own concurrency, retry, and path-confinement logic. The Kilo side is prompt text only, so it inherits whatever the host provides: real subagents, the host's own concurrency, and the permissions declared in `kilo.json`.

## Components
### Python package
- **Orchestrator** — `furrow/core/orchestrator.py`. Owns the loop.
- **Planner** — `furrow/agents/planner.py`. Returns JSON tasks (`id`, `description`, `files`, `dependencies`) plus a `rationale`. Tolerates markdown-fenced JSON via `extract_json`, accepts a bare list or a single task object, coerces ids to strings, and rejects duplicate ids and self-referential dependencies while pruning dependencies on unknown ids.
- **Worker** — `furrow/agents/worker.py`. Asks the model for `FILE: <path>` plus fenced blocks and applies them with `write_file`.
- **Tester** — `furrow/agents/tester.py`. Runs the first available test runner in the workspace, then asks the model for a JSON verdict.
- **Shared base** — `furrow/agents/base.py`. `BaseAgent` resolves the client and settings once, exposes a `model` property keyed off a per-role `model_key` (`planner_model` / `worker_model` / `tester_model`, falling back to `model`), and resolves `workspace` per call (expanduser, cwd-relative if needed, resolve). It exists because the three agents used to duplicate that preamble, and because the duplicated version passed `settings=None` straight into the client, so constructing any agent without an explicit client raised `AttributeError`.
- **LLM client** — `furrow/llm.py`. Anthropic, OpenAI, and Ollama, the last through its OpenAI-compatible API. Transient failures (rate limit, connection, timeout, 5xx) retry three times with exponential backoff; 4xx auth and validation errors do not retry, because retrying a bad API key only wastes time.
- **CLI** — `furrow start "<goal>"` and `furrow web`. Exits 130 on interrupt.
- **Web** — `furrow/web/server.py`. Serves a single inline page and streams progress over a WebSocket: `accepted`, then zero or more `progress`, then exactly one terminal `complete` or `error`. It derives progress by polling `orchestrator.cycles`, because the orchestrator exposes no event hook.

### Kilo side
- **Orchestrator agent** — `.kilo/agent/furrow.md`, the default agent in `kilo.json`. Runs the plan, execute, test, repeat cycle and delegates to the subagents below.
- **Planner subagent** — `.kilo/agent/furrow-planner.md`. Breaks the goal into tasks with the same shape the Python planner emits.
- **Worker subagent** — `.kilo/agent/furrow-worker.md`. Executes one task, and unlike the Python worker it can run tests and lint itself through its own `bash` tool.
- **Tester subagent** — `.kilo/agent/furrow-tester.md`. Runs tests, lint, and type checks and fixes what fails, rather than only reporting a verdict.
- **Command** — `.kilo/command/furrow.md`. The `/furrow` entry point, which starts the orchestrator with a goal.
- **Limits** — The Kilo loop has no numeric cycle budget and no stall threshold; it runs until the goal is met or the user stops the session.

## Loop Semantics
Each cycle: plan, execute in dependency waves, run the test suite, then route failures into the next planning prompt. The goal string is never mutated; failure summaries are held separately, prepended to the next planning prompt, and cleared on a green run.

The run is finished when:
- a non-empty plan had every task completed and the tester reported `passed`, or
- any task failed.

It halts, without necessarily meeting the goal, after any of:
- `MAX_CONSECUTIVE_FAILURES = 3` consecutive cycles that raise,
- `MAX_NO_PROGRESS_CYCLES = 3` consecutive cycles that complete no task,
- `FURROW_MAX_CYCLES` cycles, where `0` means unlimited.

History: the loop previously ran exactly one cycle and reported success, because `_is_done()` compared against a task list that was always empty. Termination, the cycle budget, the stall guards, and per-cycle exception handling were all added at the same time.

## Concurrency
`FURROW_MAX_PARALLEL_TASKS` (default 5) caps concurrent workers through an `asyncio.Semaphore`. Tasks are scheduled in dependency waves: a task waits for its dependencies, tasks within a wave run concurrently, and a task whose dependency failed or was skipped is marked `skipped` without running. Unresolvable dependencies fall back to unordered execution rather than stalling the loop.

## Safety
- The worker only writes files. It never executes shell commands authored by a model.
- Writes are confined to the workspace: absolute paths, `..` components, and paths resolving outside the root, including through symlinks, are rejected and named in the task result.
- The tester runs the project's own test command with a 120-second timeout, and kills and reaps the child on overrun rather than leaking it.
- Honest limitation: the tester infers its verdict from LLM output, so it is only as reliable as that output. Absent an explicit verdict it reports inconclusive and treats it as a failure rather than a pass.

## Files
```
kilo.json
pyproject.toml
README.md
furrow/
  __init__.py  config.py  llm.py
  agents/  __init__.py  base.py  planner.py  worker.py  tester.py  prompts.py
  core/    __init__.py  orchestrator.py
  cli/     __init__.py  main.py
  web/     __init__.py  server.py
.kilo/
  FURROW.md  .gitignore
  agent/     furrow.md  furrow-planner.md  furrow-worker.md  furrow-tester.md
  command/   furrow.md
tests/
```
Note that `kilo.json` is at the repo root, not inside `.kilo/`.

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`, declared in `kilo.json`. This is what lets the Kilo-side worker run tests and lint itself, a capability the Python package deliberately does not give a model.

## Usage
- Run `/furrow <goal>` to start the loop. Mid-loop user input is folded into the next cycle's plan.
- Run `furrow start "<goal>"` to start the same loop from the Python package, or `furrow web` for the web view.
- The two sides differ as noted above: the Kilo worker and tester can act on failures themselves, and the Kilo loop has no numeric cycle or stall thresholds.

## Future Work
- Persistent state for goal and task tracking across sessions. There is no state module anywhere in `furrow/`.
- A desktop wrapper for non-TUI interaction.
- Progress streaming richer than polling the cycle counter, meaning a real event or callback hook on the orchestrator.
- Broader test coverage. The suite is a small set of unit tests and has never been executed in this environment.
- Keeping the Python and Kilo implementations described above in sync.
