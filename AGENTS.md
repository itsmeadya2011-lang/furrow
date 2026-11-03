# AGENTS.md

This file gives Kilo agents a quick orientation to the Furrow project. It is not
a substitute for the detailed documentation in [`.kilo/FURROW.md`](.kilo/FURROW.md)
or the README.

## What is Furrow?

Furrow is an **autonomous coding agent** that plans, implements, and tests changes
to a codebase through an iterative loop. It exists in two layers that share the
same name:

1. **Kilo integration** (`.kilo/`) — agent definitions, a slash command
   (`/furrow <goal>`), and architecture docs for running Furrow inside Kilo.
2. **Python package** (`furrow/`) — a standalone CLI/orchestrator/worker/tester/web
   stack.

## Project layout

```
furrow/
├── __init__.py        # package entry; re-exports Settings and LLMClient
├── config.py          # pydantic models + Settings (FURROW_ prefix)
├── llm.py             # provider-neutral LLM client (Anthropic/OpenAI/Ollama)
├── costs.py           # per-model USD pricing and cost estimation
├── logging.py         # structlog with a stdlib fallback
├── agents/
│   ├── prompts.py     # system prompts + worker tool schemas
│   ├── parsing.py     # robust JSON extraction from LLM output
│   ├── planner.py     # PlannerAgent: goal -> Plan
│   ├── worker.py      # WorkerAgent: tool-calling loop (read/write/list/run)
│   └── tester.py      # TesterAgent: detects and runs the test suite
├── core/
│   ├── events.py      # EventBus: bounded fan-out pub/sub
│   ├── orchestrator.py# main loop: plan -> execute -> test -> repeat
│   └── state.py       # RunState + atomic StateStore persistence
├── cli/
│   └── main.py        # Click CLI: start, status, resume, web
└── web/
    └── server.py      # FastAPI app + websocket event streaming
```

## Build / test / lint

```bash
pip install -e ".[dev]"
python -m pytest -q            # 305 offline tests, no API keys needed
python -m ruff check .         # linting
python -m mypy furrow          # type checking
```

## Conventions

- Python >= 3.10. `from __future__ import annotations` everywhere.
- Line length 100 (ruff). Ruff rule set: `E, F, I, UP, B, SIM, RUF`.
- Defensive code is a feature: every untrusted input (LLM output, tool args,
  task statuses) is coerced or rejected before use, never allowed to crash the
  run. A planner/execute/tester/state-save failure emits an ERROR event and
  ends the loop cleanly instead of raising.
- The worker's command execution is hardened: commands run through
  `/bin/sh -c`, are split into segments, and every segment and command
  substitution is checked against a deny-list of destructive programs and
  patterns. Force-pushes and redirections outside the workspace are denied.
- State is persisted atomically to `<workspace>/.furrow/state.json`. That
  directory is gitignored and is local scratch state.

## Version control

- Work on the branch the session was given (`kilo/<name>`).
- Commit at the end of every cycle that produced changes. Describe the change,
  not the process.
- Push to `origin` before the session ends and set the tracking branch.
- If given a base branch (`session/agent_<uuid>`), fast-forward it to the run's
  tip and push it too.
- Never force-push or rewrite a published branch.
- Leave `.furrow/` untracked.

## Known gaps

- Currency cost *estimation* is implemented (`furrow.costs`); the "Future Work"
  section of `.kilo/FURROW.md` still says it is missing — that doc is out of
  date and should be corrected.
- No `--version` flag on the CLI.
- The tester detects pytest/npm/pnpm/yarn/cargo/go but not `uv`.