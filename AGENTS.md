# AGENTS.md

This file provides guidance for AI agents working on the Furrow codebase.

## Project Overview

Furrow is an autonomous coding agent (Python 3.10+) that plans, implements, and tests changes to a codebase through an iterative loop. It coordinates a planner, a worker (which edits files using real tool calls), and a tester, persisting state and optionally streaming progress via a web UI.

The project has two layers that share the name:
- **Python package** (`furrow/`) — installable CLI + orchestrator + web server
- **Kilo agent integration** (`.kilo/`) — primary agent, subagents, slash command running the same loop inside Kilo

## Repository Layout

```
/workspace/app
├── AGENTS.md              # this file
├── Makefile               # developer convenience targets
├── README.md              # user-facing feature/config/usage docs
├── pyproject.toml         # build, deps, lint/type/test config
├── kilo.json              # Kilo project config (default_agent: furrow)
├── .github/workflows/ci.yml
├── .gitignore
├── .kilo/                 # Kilo agent integration
│   ├── agent/
│   │   ├── furrow.md          # primary orchestrator agent
│   │   ├── furrow-planner.md  # planner subagent
│   │   ├── furrow-worker.md   # worker subagent
│   │   └── furrow-tester.md   # tester subagent
│   ├── command/
│   │   └── furrow.md          # /furrow <goal> slash command
│   └── FURROW.md            # architecture doc (both layers)
├── furrow/                # Python package (~4,500 lines)
│   ├── __init__.py        # re-exports Settings, LLMClient
│   ├── config.py          # Settings (pydantic), Provider, TaskStatus, models
│   ├── costs.py           # ModelPricing, DEFAULT_PRICING, estimate_cost
│   ├── llm.py             # LLMClient (Anthropic/OpenAI/Ollama), retries, tools
│   ├── logging.py         # structlog config w/ stdlib fallback
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── parsing.py       # JSON extraction / plan / test parsing
│   │   ├── planner.py       # PlannerAgent
│   │   ├── prompts.py       # system prompts + WORKER_TOOLS schemas
│   │   ├── tester.py        # TesterAgent + test runner detection
│   │   └── worker.py        # WorkerAgent + hardened command guard
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py        # Click CLI: start/status/resume/web
│   ├── core/
│   │   ├── __init__.py
│   │   ├── events.py        # Event, EventBus, EventType
│   │   ├── orchestrator.py  # Orchestrator loop (599 lines)
│   │   └── state.py         # RunState, HistoryEntry, StateStore (atomic)
│   └── web/
│       ├── __init__.py
│       └── server.py        # FastAPI app + inline SPA + WebSocket
└── tests/                 # ~387 test functions, all offline (no API keys)
    ├── __init__.py
    ├── test_cli.py
    ├── test_config.py
    ├── test_core.py
    ├── test_costs.py
    ├── test_events.py
    ├── test_llm.py
    ├── test_orchestrator.py
    ├── test_parsing.py
    ├── test_planner.py
    ├── test_state.py
    ├── test_tester.py
    ├── test_web.py
    └── test_worker.py
```

## How to Run

Install in editable mode with dev dependencies:

```bash
pip install -e ".[dev]"
```

Run the full offline test suite:

```bash
python -m pytest -q
```

Lint:

```bash
python -m ruff check .
```

Type check:

```bash
python -m mypy furrow
```

All tests run offline and do not require API keys.

## Key Architectural Concepts

### The core loop (one cycle)

1. **Plan** — `PlannerAgent.plan(goal, context)` → `Plan` (tasks with id, description, files, dependencies)
2. **Merge** — `Orchestrator._merge_plan()` preserves terminal task statuses and attempt counters; skips already-completed unchanged tasks
3. **Execute** — `_plan_waves()` computes dependency waves; workers run concurrently up to `max_parallel_tasks`, each with a real tool-calling loop (`read_file`, `write_file`, `list_files`, `run_command`)
4. **Test** — `TesterAgent.run()` detects the project's test runner (pytest/npm/pnpm/yarn/cargo/go), runs it, returns `TestResult`
5. **Persist & emit** — events emitted throughout; `RunState` saved atomically to `<workspace>/.furrow/state.json`

### Key data models

- `TaskModel` — id, description, files, dependencies, status, result, error, attempts, timestamps, files_changed
- `Plan` — tasks list + rationale
- `TestResult` — passed, summary, failures, skipped
- `WorkerResult` — summary, success, files_changed, error, iterations
- `RunResult` — passed, cycles, tasks_completed/failed/blocked, tests_run, summary, stopped_reason, usage, files_changed
- `Usage` — input_tokens, output_tokens, calls, cost_usd, total_tokens property
- `Settings` — pydantic BaseSettings with `FURROW_` env prefix, bounded numeric fields

### Command execution security (worker.py)

Commands run through `/bin/sh -c` in their own process group. A hardened guard:
- Normalizes (lowercases, collapses whitespace, canonicalises `$HOME`)
- Splits into segments on `;`, `&&`, `||`, lone `&`, `|`, newlines
- Checks every segment and every command/process substitution
- Token-aware `rm`/`chmod` guard (flags in any order, `--` terminator, wrappers like `sudo`/`env`/`nohup`)
- Denies destructive programs by name and destructive patterns
- Denies force pushes (`git push --force`, `-f`, `--force-with-lease`, `+ref` refspecs)
- Denies redirections whose target resolves outside the workspace

## Coding Conventions

- **Python 3.10+** syntax (`from __future__ import annotations` on every module)
- **Type hints** on all function signatures; use `X | None` union syntax
- **Docstrings** on public classes and functions (PEP 257 style, one-line summary)
- **Line length 100** (ruff config)
- **Defensive coding**: `getattr` for provider SDK attributes, `try/except Exception` with `# pragma: no cover - defensive` for non-core paths, never raise from cost estimation or LLM parsing
- **Async-first**: the orchestrator/worker/tester/web stack is `asyncio`-based
- **No `conftest.py`** — fixtures are defined inline per test module
- Tests are **offline** — LLM clients are faked/monkeypatched, no API keys required

## Files to Modify for Common Tasks

| Task | Files |
|------|-------|
| Add a new LLM provider | `furrow/llm.py` (Provider enum, `_complete_*`, `_chat_*`) |
| Add a new test runner | `furrow/agents/tester.py` (`detect_command`) |
| Add a new CLI command | `furrow/cli/main.py` |
| Add a new API endpoint | `furrow/web/server.py` |
| Change the core loop | `furrow/core/orchestrator.py` |
| Change state persistence | `furrow/core/state.py` |
| Change pricing table | `furrow/costs.py` (`DEFAULT_PRICING`) |
| Change settings | `furrow/config.py` (`Settings`) |

## Kilo Integration

The Kilo-side agents and command live in `.kilo/`:
- `.kilo/agent/furrow.md` — primary orchestrator (50 steps, loops forever)
- `.kilo/agent/furrow-planner.md` — planner subagent (returns JSON plan)
- `.kilo/agent/furrow-worker.md` — worker subagent (executes one task)
- `.kilo/agent/furrow-tester.md` — tester subagent (returns pass/fail)
- `.kilo/command/furrow.md` — `/furrow <goal>` slash command
- `kilo.json` — sets `default_agent: "furrow"`, permissions `bash/edit/read: allow`

## Git / Version Control

- Work on the branch the session was given. Do not force-push or rebase published branches.
- Commit at the end of every cycle that produced changes.
- Push the branch to `origin` before the session ends.
- Leave `.furrow/` (run state) untracked — it is local scratch state, covered by `.gitignore`.
