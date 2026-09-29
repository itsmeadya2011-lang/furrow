# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects. It decomposes goals into parallelizable tasks, executes them, tests, and repeats until the goal is complete or the user stops.

## Dual Implementations
Two forms share the same five roles (orchestrator, planner, worker, tester, entry point):

- **Python package (`furrow/`)** — Standalone CLI + HTTP API; runs outside the Kilo TUI.
- **Kilo TUI plugin (`.kilo/`)** — Native Kilo integration via agents/subagents and a slash command.

## Components
- **Orchestrator** — `furrow/core/orchestrator.py` · `.kilo/agent/furrow.md` (primary). Plan → parallel workers → tester → report → repeat.
- **Planner** — `furrow/agents/planner.py` · `furrow-planner.md`. Breaks a goal into 1-5 independent tasks; emits `Plan` with `tasks[]` (`id`, `description`, `files`, `dependencies`).
- **Worker** — `furrow/agents/worker.py` · `furrow-worker.md`. Executes one assigned task; minimal, targeted changes.
- **Tester** — `furrow/agents/tester.py` · `furrow-tester.md`. Runs tests, lint, type checks; returns `TestResult` (`passed`, `summary`, `failures`).
- **Entry points** — CLI: `furrow start "<goal>"` and `furrow web` (FastAPI + Uvicorn WebSocket UI, :8000). TUI: `/furrow <goal>` (`.kilo/command/furrow.md`).

## Files
```
.kilo/agent/furrow.md          Orchestrator (primary agent)
.kilo/agent/furrow-planner.md  Planner subagent
.kilo/agent/furrow-worker.md   Worker subagent
.kilo/agent/furrow-tester.md   Tester subagent
.kilo/command/furrow.md        /furrow slash command
.kilo/FURROW.md                This file
furrow/cli/main.py             Click CLI (start, web)
furrow/core/orchestrator.py    Orchestrator + loop
furrow/agents/planner.py       PlannerAgent
furrow/agents/worker.py        WorkerAgent
furrow/agents/tester.py        TesterAgent
furrow/agents/prompts.py       Shared prompts
furrow/config.py               Pydantic settings + models
furrow/llm.py                  LLMClient (Anthropic/OpenAI/Ollama)
furrow/web/server.py           FastAPI server + WebSocket UI
kilo.json                      Project config (default_agent=furrow, open perms)
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Environment Variables
Keys fall back to bare names (`ANTHROPIC_API_KEY`/`OPENAI_API_KEY`) when `FURROW_*`-prefixed variants are unset.

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `OLLAMA_BASE_URL` | Ollama endpoint (provider=ollama) | `localhost:11434` |
| `FURROW_PROVIDER` | `anthropic`/`openai`/`ollama` | `anthropic` |
| `FURROW_MODEL` (and `_PLANNER`/`_WORKER`/`_TESTER`) | LLM per role | `claude-…` |
| `FURROW_MAX_PARALLEL_TASKS` | Concurrent workers | `5` |
| `FURROW_MAX_CYCLES` | Max cycles (`0`=unlimited) | `0` |
| `FURROW_WORKSPACE` | Working directory | cwd |
| `FURROW_LOG_LEVEL` | Log verbosity | `INFO` |
| `FURROW_LINT_ENABLED` / `_TYPE_CHECK_ENABLED` | Lint/type-check between cycles | `true` |
| `FURROW_TEST_TIMEOUT` | Test/lint timeout (s) | `120` |

## Usage
- **TUI:** `/furrow <goal>` in the Kilo terminal.
- **CLI:** `furrow start "<goal>"` or `furrow web`.
- Runs until the goal is complete or the user stops (Ctrl+C / close session).
- Mid-loop input is folded into the next cycle's plan.

## Development
```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy -p furrow
```

## Future Work
- [x] CLI wrapper for non-TUI interaction (`furrow start`)
- [x] Web UI for monitoring parallel agent activity (`furrow web`)
- [ ] Desktop wrapper for non-TUI interaction
- [ ] Persistent state file for goal/task tracking across sessions
