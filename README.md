# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow web
```

Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or run a local Ollama server
(`FURROW_PROVIDER=ollama`).

## Configuration

Settings can be set via environment variables (prefix `FURROW_`) or a `.env` file:

| Variable                  | Default                        | Description                              |
|---------------------------|--------------------------------|------------------------------------------|
| `FURROW_PROVIDER`         | `anthropic`                    | `anthropic`, `openai`, or `ollama`        |
| `FURROW_MODEL`            | `claude-sonnet-4-20250514`     | Default LLM model                         |
| `FURROW_PLANNER_MODEL`    | `claude-3-5-haiku-20241022`    | Model used by the planner agent           |
| `FURROW_WORKER_MODEL`     | `claude-3-5-sonnet-20241022`   | Model used by worker agents               |
| `FURROW_TESTER_MODEL`     | `claude-3-5-sonnet-20241022`   | Model used by the tester agent            |
| `FURROW_OLLAMA_BASE_URL`  | `http://localhost:11434`       | Ollama server URL                          |
| `FURROW_MAX_PARALLEL_TASKS` | `5`                          | Max concurrent worker tasks per cycle     |
| `FURROW_MAX_CYCLES`       | `0`                            | Max plan→execute→test cycles (0 = unbounded) |

## Architecture

Each cycle, Furrow:

1. **Plans** the goal into 1-5 parallel tasks via the planner agent.
2. **Executes** tasks in dependency-ordered waves, limited by
   `max_parallel_tasks`.
3. **Tests** the result by running the project's test suite (pytest, npm, cargo, go, etc.).
4. If tests fail, the failure context is fed back into the next plan.
5. Loops until the goal is met, the stop signal is received, or
   `max_cycles` is reached.

The orchestrator emits structured events (`cycle_start`, `plan_ready`,
`task_start`, `task_complete`, `task_failed`, `test_complete`, `cycle_end`,
`done`) for real-time UI integration.

## Programmatic Use

```python
import asyncio
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import FurrowState, StateStore

async def main():
    store = StateStore(".kilo/furrow-state.json")

    def on_event(name, data):
        print(name, data)

    orch = Orchestrator(
        goal="Add user authentication",
        on_event=on_event,
    )
    await orch.run()

    store.save(FurrowState(
        goal=orch.goal,
        cycles=orch.cycles,
        created_at="...",
        updated_at="...",
    ))

asyncio.run(main())
```

