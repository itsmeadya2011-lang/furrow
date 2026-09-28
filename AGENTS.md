# AGENTS.md

Guidance for autonomous agents working on the Furrow codebase.

## Project Overview
Furrow is an autonomous coding agent with an infinite parallel development loop
(plan -> execute in parallel -> test -> repeat). It is distributed as a Python 3.10+
package (`furrow/`) with a Click CLI and an optional FastAPI web UI.

## Layout
- `furrow/agents/` — `PlannerAgent`, `WorkerAgent`, `TesterAgent` + shared `prompts.py`
- `furrow/core/orchestrator.py` — `Orchestrator`: the plan/execute/test loop
- `furrow/llm.py` — `LLMClient`: async Anthropic/OpenAI/Ollama client with retry
- `furrow/config.py` — pydantic settings (`FURROW_` env prefix, `.env` file)
- `furrow/cli/main.py` — `furrow start` / `furrow web`
- `furrow/web/server.py` — FastAPI + WebSocket UI
- `.kilo/agent/furrow*` — Kilo TUI agent definitions (orchestrator + 3 subagents)
- `.kilo/command/furrow.md` — `/furrow` slash command

## Install (dev)
```bash
pip install -e .[dev]
```

## Test
```bash
pytest -q
```

## Lint & Type Check
```bash
ruff check .
ruff format --check .
mypy furrow
```

## Notes for Agents
- Never commit `.pyc` / `__pycache__`; the root `.gitignore` covers Python artifacts.
- Keep worker changes minimal and targeted (no scope creep).
- The orchestrator runs until all tasks complete AND tests pass, or until `max_cycles`
  is reached (0 = unlimited). Failed workers are retried once with fix instructions.
- Providers: `anthropic`, `openai`, `ollama`. Ollama uses `FURROW_OLLAMA_BASE_URL`
  and an optional `FURROW_OLLAMA_API_KEY`.
