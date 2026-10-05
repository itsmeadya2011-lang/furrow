# Furrow

Furrow is an autonomous coding agent that drives a plan -> execute -> test loop against a goal, running independent tasks in parallel until the goal is met or a stop condition is reached.

## The loop

Each cycle does three things:

1. **Plan.** The planner agent turns the goal into 1-5 tasks, each with an `id`, a `description`, a `files` hint list, and a `dependencies` list.
2. **Execute.** Tasks are grouped into dependency levels. Each level runs concurrently (bounded by `FURROW_MAX_PARALLEL_TASKS`); a task never starts before its dependencies have completed.
3. **Test.** The tester agent runs the project's test command and turns the exit code plus the output into a pass/fail verdict. A failing run feeds its failure list back into the next cycle's planner prompt as repair context, and the loop repeats.

The Python worker agent asks the model to implement one task and returns the model's summary as the task result; it does not itself apply file edits.

## Install

```bash
pip install furrow
```

To work on furrow itself, install it in editable mode with the development extras:

```bash
pip install -e ".[dev]"
```

The `dev` extra (`pyproject.toml`) provides pytest, pytest-asyncio, ruff, and mypy.

## Configuration

Configuration comes from `FURROW_*` environment variables and from a `.env` file in the working directory, both handled by `pydantic-settings` with the env prefix `FURROW_`.

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | Which provider to call: `anthropic`, `openai`, or `ollama`. |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model for completions that do not name one. |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the planner agent. |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the worker agent. |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the tester agent. |
| `FURROW_ANTHROPIC_API_KEY` | unset | Anthropic API key. |
| `FURROW_OPENAI_API_KEY` | unset | OpenAI API key. |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL of the local Ollama server. |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Maximum tasks executed concurrently inside one dependency level. |
| `FURROW_MAX_CYCLES` | `0` | Cycle limit. See below. |
| `FURROW_WORKSPACE` | current working directory | Root the file walk and the test command run against. |
| `FURROW_LOG_LEVEL` | `INFO` | Declared log level. No component reads it yet. |
| `FURROW_TEST_COMMAND` | unset (auto-detect) | Explicit shell command used to run the test suite. |
| `FURROW_MAX_TOKENS` | `4096` | Upper bound on tokens requested per Anthropic and OpenAI completion. |
| `FURROW_IGNORED_DIRS` | `.git`, `__pycache__`, `node_modules`, `.venv`, `venv`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `dist`, `build`, `target`, `.next`, `.tox`, `htmlcov` | Directories pruned while discovering project files. |

`FURROW_MAX_CYCLES` is `0` by default, and **0 or any negative value means unlimited cycles**: the loop is bounded only by its other stop conditions.

### Providers

`anthropic`, `openai`, and `ollama` are all implemented.

- **ollama** needs no API key. It posts to `/api/chat` on the server at `FURROW_OLLAMA_BASE_URL` with a 120s client timeout.
- **anthropic** and **openai** need a key. The key is read from `FURROW_ANTHROPIC_API_KEY` / `FURROW_OPENAI_API_KEY`, and if those are unset the client falls back to the bare `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` environment variables. Passing `--provider` on the command line makes furrow check for the key up front and fail with an actionable usage error rather than a late `ValueError`.

Provider calls are retried up to 3 attempts with exponential backoff (~1s then ~2s, capped at 10s), except for configuration errors and cancellation.

## CLI

```bash
furrow start "<goal>" [--model M] [--provider {anthropic,openai,ollama}] \
                 [--max-cycles N] [--max-parallel N] [--test-command CMD]
furrow web [--host H] [--port P]
```

`furrow start` runs the loop in the terminal. The goal argument is optional; without it, furrow prompts for one. Omitting `--max-cycles` means unlimited cycles. CLI options override the corresponding settings for that invocation only; the process-wide settings object is left untouched.

`furrow web` serves the browser UI. `--host` defaults to `0.0.0.0` and `--port` to `8000`.

## Termination

The loop halts when any of the following is true (`furrow/core/orchestrator.py`):

1. The planner returns a plan with no tasks.
2. Every task completed and the tests passed.
3. The number of consecutive failing cycles exceeds `MAX_CONSECUTIVE_FAILURES` (3). This is the bound that stops an otherwise unlimited run.
4. `FURROW_MAX_CYCLES` is positive and has been reached.

A cycle that raises is counted as a failed cycle rather than ending the run, and the console reports it before the next cycle starts.

## Task dependencies

The planner emits a `dependencies` list per task. The orchestrator schedules tasks in dependency levels: every task whose dependencies have completed forms one level, levels run in order, and the tasks inside a level run concurrently under the `FURROW_MAX_PARALLEL_TASKS` bound. A task that lists an unknown dependency id, depends on itself, or sits in a dependency cycle can never run: it is marked failed and reported as blocked with the reason.

## Test command detection

Without `FURROW_TEST_COMMAND` / `--test-command`, furrow tries these runners in order:

```
pytest -q
python -m pytest -q
npm test -- --silent
pnpm test -- --silent
yarn test --silent
cargo test -q
go test ./...
```

A candidate is skipped only when its binary cannot be spawned at all, or when it hangs past the timeout. The first candidate that runs to completion is authoritative: its real exit code decides pass/fail, so a red suite is reported as a failure instead of being hidden by falling through to another runner. The exit code is applied after the model's verdict and always wins: a non-zero exit code or a timeout is never reported as a pass.

Each run is killed after 120 seconds. A timed-out run, an unspawnable configured command, and a workspace with no spawnable runner are all reported as failures, never as passes.

## Web UI

`furrow web` serves a single-page UI at `http://<host>:<port>`.

- **`/`** returns the page: a goal field, a Start button, a Stop button, and a live event log.
- **`/ws`** (websocket) streams one orchestrator run. The client sends `{"goal": "...", "model": "..."}` to start and `{"type": "stop"}` at any time to halt the run. The server sends one JSON object per orchestrator event and always ends with `{"type": "stopped", "reason": ...}`. The Stop button cancels the running orchestrator task.
- **`GET /health`** returns `{"status": "ok", "active_runs": N}`.
- **`POST /stop`** requests a stop, for the run named by the `?run_id=` query parameter or, by default, for every active run.

Events are buffered on an unbounded queue and flushed by a dedicated sender task, so the orchestrator never blocks on a slow browser and event order is preserved.

## Development

`pyproject.toml` configures pytest (`asyncio_mode = "auto"`, `testpaths = ["tests"]`), ruff (line length 100, lint rules `E`, `F`, `I`), and mypy (Python 3.10, typed defs required).

```bash
pytest
ruff check .
mypy furrow
```
