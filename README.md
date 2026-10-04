# Furrow

An autonomous coding agent. You give it a goal; it plans the work, implements it
in parallel, runs the test suite, feeds the failures back into the next round,
and stops when the goal is met.

## How the loop actually behaves

Each cycle is **plan → execute → test**:

1. **Plan** — a planner model turns the goal into 1–5 tasks, each with the files
   it touches and the task ids it depends on.
2. **Execute** — a worker model implements each task. Tasks run concurrently,
   bounded by `max_parallel_tasks`, and a task waits for its dependencies. A task
   whose dependency failed is **skipped**, not hung; an unresolvable dependency
   cycle is skipped too.
3. **Test** — the project's own test runner is executed in the workspace and a
   model judges the output.
4. **Repair** — a failing suite is prefixed onto the goal and the loop cycles
   again with the failures in view. Completed tasks are re-opened rather than
   discarded.

**It is bounded, not literally infinite.** Despite the branding, the loop halts
when any of these is true:

| Bound | Value | Where |
| --- | --- | --- |
| Cycle budget | `max_cycles` (`0` = unlimited, the default) | `furrow start --max-cycles` / `FURROW_MAX_CYCLES` |
| No-progress guard | `Orchestrator.MAX_UNPRODUCTIVE_CYCLES = 3` consecutive cycles that complete no task for the first time | class attribute |
| Repair guard | `Orchestrator.MAX_REPAIR_ATTEMPTS = 3` consecutive repair cycles that leave the suite failing | class attribute |

The guards matter because the default cycle budget *is* unlimited. A permanently
red suite would otherwise loop and bill indefinitely; instead it stops after
three failed repair attempts and reports why. The reason is always available in
`RunResult.stopped_reason`.

Failures are contained rather than fatal: a planner that returns malformed JSON
(one repair request is made), a provider error, or a worker exception is recorded
and the next cycle proceeds. Only cancellation escapes the loop.

## Requirements

- Python 3.10 or newer
- An API key for the provider you choose (Anthropic or OpenAI), or a local
  [Ollama](https://ollama.com) server

## Install

```bash
pip install furrow
```

## Quick start

There are three ways in. The Kilo/TUI path is the primary product surface; the
CLI and the web server are thin wrappers over the same orchestrator.

### 1. As a Kilo agent (primary)

Furrow ships four Kilo agent definitions and a slash command under `.kilo/`, and
`kilo.json` sets `default_agent: "furrow"` with open permissions (`bash`, `edit`
and `read` all `allow`). Start the loop from the TUI:

```
/furrow Add JWT authentication to the API
```

`/furrow` invokes the `furrow` orchestrator agent, which plans, spawns
`furrow-worker` subagents, tests, and repeats. The four agents and the loop's
bounds are documented in [`.kilo/FURROW.md`](.kilo/FURROW.md).

### 2. Command line

```bash
furrow start "Add JWT authentication to the API"
```

| Flag | Effect |
| --- | --- |
| `--provider [anthropic\|openai\|ollama]` | Override the configured provider |
| `--model TEXT` | Use one model for every role, overriding the per-role models |
| `--workspace TEXT` | Directory to work in (must exist) |
| `--max-parallel INTEGER` | Maximum tasks in parallel (≥ 1) |
| `--max-cycles INTEGER` | Maximum cycles; `0` means unlimited |
| `--test-command TEXT` | Test command; auto-detected when unset |
| `--quiet` | Suppress the banner, panels and spinner |

`furrow --version` prints the version. The goal argument is optional; omitting it
prompts for one.

### 3. Web UI

```bash
furrow web --host 127.0.0.1 --port 8000 --reload
```

| Flag | Default | Effect |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Interface to bind |
| `--port` | `8000` | TCP port |
| `--reload` | off | Reload on code changes (development only) |

## Configuration

Copy [`.env.example`](.env.example) to `.env` — it is loaded automatically by
`pydantic-settings`, is the source of truth for the list below, and is not
tracked in git. Every setting can also be set as a `FURROW_`-prefixed environment
variable.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | `anthropic`, `openai` or `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model for general completion |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model for the planning stage |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model for the execution stage |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model for the testing stage |
| `FURROW_OVERRIDE_MODEL` | unset | If set, wins over all four role models |
| `FURROW_ANTHROPIC_API_KEY` | unset | Falls back to `ANTHROPIC_API_KEY` |
| `FURROW_OPENAI_API_KEY` | unset | Falls back to `OPENAI_API_KEY` |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL of the local Ollama service |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Maximum tasks executed in parallel |
| `FURROW_MAX_CYCLES` | `0` | Cycle budget; `0` means unlimited |
| `FURROW_MAX_TOKENS` | `4096` | Max tokens per LLM request |
| `FURROW_REQUEST_TIMEOUT` | `120.0` | HTTP timeout in seconds for LLM requests |
| `FURROW_MAX_RETRIES` | `3` | Retries on transient LLM failures |
| `FURROW_TEST_TIMEOUT` | `120.0` | Seconds before a test command is killed |
| `FURROW_TEST_COMMAND` | unset | Explicit test command; auto-detected when unset |
| `FURROW_WORKSPACE` | current working directory | Project directory, and the filesystem sandbox root |
| `FURROW_LOG_LEVEL` | `INFO` | Log verbosity |

## Providers

| Provider | Key env var | Notes |
| --- | --- | --- |
| `anthropic` | `ANTHROPIC_API_KEY` or `FURROW_ANTHROPIC_API_KEY` | Default |
| `openai` | `OPENAI_API_KEY` or `FURROW_OPENAI_API_KEY` | |
| `ollama` | none | Local server at `http://localhost:11434` by default; the model must already be pulled |

Retries with exponential backoff apply to transient provider failures only —
rate limits, timeouts, dropped connections and 5xx responses. 4xx
authentication and validation errors are deterministic, so they are not retried.

## Exit codes

`furrow start` reports the outcome through its exit code:

| Code | Meaning |
| --- | --- |
| `0` | The run reported `RunResult.completed` |
| `1` | The run finished without completing the goal (failing tasks or tests, exhausted budget, or an error) |
| `2` | Configuration was rejected before any work started |
| `130` | Interrupted with `Ctrl-C` |

## Web UI

`furrow web` serves a single-page UI backed by a FastAPI app:

| Route | Behaviour |
| --- | --- |
| `GET /` | The bundled single-page UI (no build step) |
| `GET /health` | Liveness probe; returns `{"status": "ok"}` |
| `POST /api/start` | Schedules a run and returns `202` with a run id immediately |
| `WS /ws` | Accepts `{"goal": ..., "model": ...}` frames and streams JSON event frames back |

`/ws` carries any number of runs on one connection, serialised. Each frame of a
run is a `FurrowEvent` plus a `run_id`; the last is a `run_end` frame carrying the
`RunResult`. Closing the socket cancels the run, so a closed tab stops the API
spend.

### Security

> **Furrow has no authentication.** It is an agent that writes code and shells
> out to your test runner. The default bind is `127.0.0.1`. Binding anything else
> — `0.0.0.0`, a LAN address, a container bridge — publishes unauthenticated
> remote code execution to anyone who can reach the port. A loud warning is
> printed to stderr for a non-loopback bind, but the exposure is your decision to
> make. Use a tunnel or keep it on loopback unless you have arranged your own
> access control.

## Python API

`Orchestrator` is the integration point: give it a goal and an `on_event`
callback, and drive it yourself.

```python
import asyncio

from furrow import LLMClient, Settings
from furrow.core import EventType, Orchestrator


async def main() -> int:
    settings = Settings(override_model="claude-3-5-sonnet-20241022", max_cycles=4)
    client = LLMClient(settings=settings)

    def on_event(event) -> None:
        if event.type is EventType.TASK_COMPLETE:
            print(f"cycle {event.cycle}: {event.task_id} done")

    orchestrator = Orchestrator(
        goal="Add a health check endpoint",
        client=client,
        settings=settings,
        on_event=on_event,
    )
    result = await orchestrator.run()
    print(result.completed, result.stopped_reason)
    for task in orchestrator.tasks:
        print(task.id, task.status)
    return 0 if result.completed else 1


asyncio.run(main())
```

`on_event` is called from async context, so it must not block; exceptions it
raises are swallowed rather than breaking the run. Available event types are
`run_start`, `cycle_start`, `plan`, `task_start`, `task_complete`, `task_failed`,
`task_skipped`, `test_start`, `test_result`, `cycle_end`, `error` and `run_end`.
Task statuses are `pending`, `running`, `completed`, `failed` and `skipped`.

Everything else is importable from the package root: `LLMClient`, `Settings`,
`Provider`, `TaskStatus`, `TaskModel`, `Plan`, `TestResult`, `get_settings`.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check .
mypy furrow
```

The dev extra is `pytest` (with `pytest-asyncio` in auto mode), `ruff` and
`mypy`. Ruff enforces a 100-character line length targeting Python 3.10; mypy
runs over the package and excludes `tests/`.
