# Furrow

Autonomous coding agent that plans, executes and verifies work in a loop.

## Features

- **Goal-directed loop.** Give Furrow a goal and it plans, executes and tests, then immediately starts the next cycle with what it learned.
- **Specialised agent roles.** A planner breaks the goal into tasks, workers execute individual tasks, and a tester verifies the result.
- **Dependency-ordered parallelism.** Tasks run in waves: a task starts only once its declared dependencies have completed, with at most `FURROW_MAX_PARALLEL_TASKS` workers running at the same time.
- **Honest failure statuses.** A task whose dependency failed is marked `BLOCKED`, and a plan whose dependencies form a cycle is detected and blocked instead of hanging.
- **Verification gate.** Each cycle that plans work ends with a test run. The subprocess exit code is authoritative — the model only summarises the output and extracts per-failure detail.
- **Bounded, self-terminating loop.** The run halts on a real termination condition instead of cycling forever; see "Loop termination".
- **Terminal UI.** Cycles, plans, task results and test output are rendered with `rich`.
- **CLI and web monitor.** `furrow start` runs the loop in your terminal; `furrow web` serves the same loop as a small FastAPI app in the browser.

## Requirements

- Python 3.10 or newer
- An API key for the provider you select. `anthropic` and `openai` need one; `ollama` talks to a local server and needs none.

## Install

Install from a checkout of this repository:

```bash
pip install -e ".[dev]"
```

That is the only supported install path. This project is not published to PyPI, and the name `furrow` on PyPI belongs to an unrelated text-formatting package, so `pip install furrow` installs something else entirely. The `dev` extra adds `pytest`, `pytest-asyncio`, `ruff` and `mypy`.

The build backend is `setuptools.build_meta` (`requires = ["setuptools>=68", "wheel"]`) and `requires-python` is `>=3.10`, so no build tooling needs installing by hand. The console script `furrow` is declared in `[project.scripts]`.

## Configuration

Every setting is defined on `Settings` in `furrow/config.py`, uses the `FURROW_` environment prefix, and is also read from a `.env` file in the working directory.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | Provider to call: `anthropic`, `openai` or `ollama`. |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Model used for a completion when the caller names none. Under `ollama` this is the model name requested from the local server. |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the planner. |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by workers. |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the tester. |
| `FURROW_ANTHROPIC_API_KEY` | unset | Key for the Anthropic API. |
| `FURROW_OPENAI_API_KEY` | unset | Key for the OpenAI API. |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL of the Ollama server. |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Upper bound on workers running concurrently inside one wave. |
| `FURROW_MAX_CYCLES` | `0` | Stop after this many cycles; `0` means no cycle limit. |
| `FURROW_MAX_RETRIES` | `2` | Extra attempts for a retryable LLM error, and extra re-asks when the planner's JSON cannot be parsed (capped at 3). |
| `FURROW_TEST_TIMEOUT` | `120` | Seconds before the test subprocess is killed and reaped. |
| `FURROW_TEST_COMMAND` | unset | Explicit test command; when unset a runner is auto-detected. |
| `FURROW_MAX_IDLE_CYCLES` | `3` | Stop after this many consecutive cycles that made no progress; `0` means no idle limit. |
| `FURROW_WORKSPACE` | the current directory | Working directory for the test subprocess and for resolving relative task file paths. |
| `FURROW_LOG_LEVEL` | `INFO` | Defined on `Settings` but not read by any code yet; there is no logging setup for it to configure. |

Notes on these settings:

- Raw `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` environment variables are honoured as a fallback for `FURROW_ANTHROPIC_API_KEY` and `FURROW_OPENAI_API_KEY`.
- `FURROW_MAX_PARALLEL_TASKS` has no "unlimited" setting: a value of `0` or less is clamped to 1, so the wave still runs serially. Only the two cycle counters treat `0` as unlimited.
- `FURROW_MAX_RETRIES` does double duty. `LLMClient.complete` retries a transient provider error (HTTP 408, 409, 425, 429, any 5xx, or an exception class named like a timeout/rate-limit/overload) with exponential backoff, up to `max_retries + 1` attempts. `PlannerAgent.plan` separately re-asks the model when the response cannot be parsed as JSON. Configuration errors such as a missing API key are not retried.
- **`FURROW_TEST_COMMAND` is the reliable way to pin the verification gate.** With it unset, `TesterAgent` walks a fixed candidate list in priority order — `pytest -q`, `python -m pytest -q`, `python3 -m pytest -q`, `npm test -- --silent`, `pnpm test -- --silent`, `yarn test --silent`, `cargo test -q`, `go test ./...` — and uses the first candidate that can actually run. That first usable runner wins even if it later fails, so an unrelated toolchain on `PATH` can capture the gate. Setting `test_command` removes the ambiguity: the command is authoritative, detection is skipped entirely, and the exit code you get is the one that decides pass/fail.

## Usage

```bash
furrow start "Add JWT auth to the API"
```

Running `furrow start` with no goal prompts for one interactively. `furrow --version` prints the installed version.

### `furrow start`

| Option | Default | Effect |
| --- | --- | --- |
| `GOAL` (argument) | prompted | The objective for the run. |
| `--model TEXT` | unset | Model for every agent. |
| `--provider [anthropic\|openai\|ollama]` | `anthropic` | LLM provider. |
| `--max-cycles INTEGER` | `0` | Stop after this many cycles; `0` is unlimited. |
| `--max-parallel-tasks INTEGER` | `5` | Tasks to run concurrently in one wave. |
| `--max-idle-cycles INTEGER` | `3` | Stop after this many consecutive no-progress cycles; `0` is unlimited. |
| `--workspace PATH` | the current directory | Directory the agents work in. |
| `--test-command TEXT` | unset | Test command; auto-detected when omitted. |

`--model` sets `model`, `planner_model`, `worker_model` and `tester_model` together. This matters: each agent is handed its own model by the orchestrator, so overriding only `settings.model` would leave the planner, workers and tester on their configured defaults and the flag would silently do nothing.

```bash
furrow start --model claude-sonnet-4-20250514 --max-cycles 10 --max-parallel-tasks 3 "Add JWT auth to the API"
```

### `furrow web`

| Option | Default | Effect |
| --- | --- | --- |
| `--host TEXT` | `0.0.0.0` | Interface to bind. |
| `--port INTEGER` | `8000` | Port to bind. |

```bash
furrow web --host 127.0.0.1 --port 8080
```

### Local models with Ollama

`Provider.OLLAMA` is routed through Ollama's OpenAI-compatible API, so a local server works with no API key. Two things to know:

- The client is built from `FURROW_OLLAMA_BASE_URL` with a placeholder key, and the model name is always `FURROW_MODEL`. `FURROW_PLANNER_MODEL`, `FURROW_WORKER_MODEL` and `FURROW_TESTER_MODEL` are ignored under this provider, because a local server serves the one model you ask for. Set `FURROW_MODEL` to a name you have actually pulled.
- `--model` does reach the Ollama request, since it sets `model` too.

```bash
export FURROW_PROVIDER=ollama
export FURROW_OLLAMA_BASE_URL=http://localhost:11434
export FURROW_MODEL=llama3
furrow start "Add JWT auth to the API"
```

## Loop termination

The loop is not unconditional. After each cycle `Orchestrator._is_done()` halts the run when any of these holds:

1. `FURROW_MAX_CYCLES` is greater than zero and has been reached.
2. The planner returned an empty task list.
3. The planner set `complete: true` on the plan **and** the test suite is not known to be red. A missing test signal is tolerated, so a project with no runner can still finish; a known-failing suite always wins over the planner's claim.
4. `FURROW_MAX_IDLE_CYCLES` consecutive cycles made no progress. A cycle is idle when nothing was executed, when the planner declared the goal complete, when planning failed, or when the plan was empty.

`run()` re-checks the cycle cap after `_is_done()` as a backstop, then prints a summary panel with cycle count, per-status task counts, the last test verdict and the idle-cycle count. An exception inside a cycle is reported and counted as an idle cycle rather than ending the run. Ctrl+C prints a stop message and exits 0.

The planner sees a focus string rebuilt each cycle: the original goal (never mutated), a bounded list of already-completed task summaries, and the current test failures. That is what keeps the loop working the same objective instead of chasing the last failure.

## Architecture

| Path | Responsibility |
| --- | --- |
| `furrow/core/orchestrator.py` | The loop: plan, execute in dependency-ordered waves, test, repeat, and decide when to stop. |
| `furrow/agents/planner.py` | Turns a goal into a `Plan`, tolerating fenced or prose-wrapped JSON. |
| `furrow/agents/worker.py` | Executes one assigned task and returns a summary. |
| `furrow/agents/tester.py` | Runs the test command, then asks the model to summarise it. The exit code decides pass/fail. |
| `furrow/agents/prompts.py` | System prompts for the three agents above. |
| `furrow/llm.py` | Async LLM client (Anthropic, OpenAI, Ollama) with retries, plus async file helpers. |
| `furrow/config.py` | `Settings`, plus the `Plan`, `TaskModel`, `TaskStatus` and `TestResult` models. |
| `furrow/cli/main.py` | The `furrow` command-line entry point. |
| `furrow/web/server.py` | FastAPI app backing `furrow web`. |

Two details the rest of the package leans on:

- `Orchestrator` takes an optional `on_event: Callable[[str], None]` hook. Every message the loop prints is also passed to it, with rich renderables flattened to plain text first. A consumer that raises is logged and ignored. The web monitor streams over this hook and strips rich markup before it reaches the browser.
- The CLI mutates the global `settings` singleton in place; the web app instead builds a per-run `Settings` copy so concurrent runs cannot disturb each other.

This repository ships **two parallel implementations** of the same agent architecture. The Python package above is one. The other is a set of Kilo markdown agents under `.kilo/agent/`, driven by the `/furrow` slash command in `.kilo/command/furrow.md`, which runs the same plan/execute/test loop through Kilo's own agent runtime. `.kilo/FURROW.md` documents the markdown-agent side; this README documents the Python package.

## Web UI

`furrow web` serves a FastAPI app with three routes:

| Route | Purpose |
| --- | --- |
| `GET /` | The single-page monitor. `?goal=` pre-fills the form (HTML-escaped). |
| `GET /health` | Liveness probe; returns `{"status": "ok"}`. |
| `WS /ws` | The browser sends `{"goal": "...", "model": "..."}` and receives loop output as text frames until the run ends. |

The page derives its socket URL from `location.protocol`, using `wss://` under HTTPS and `ws://` otherwise, so serving the app behind a TLS-terminating proxy needs no configuration change. `model` in the payload overrides all four model settings for that run only.

The server has no authentication and a run drives an LLM against the workspace on this machine, so bind it to loopback unless you mean otherwise:

```bash
furrow web --host 127.0.0.1 --port 8000
```

The default `--host 0.0.0.0` is exposed to the whole network, and the server prints a warning when it is bound to any non-loopback host. Closing the browser tab cancels the run instead of leaving it burning tokens.

## Dependencies

Runtime dependencies, as pinned in `pyproject.toml`:

| Package | Role |
| --- | --- |
| `fastapi`, `uvicorn[standard]` | The web monitor. `furrow start` never loads them; the `web` command imports the app inside the command body. |
| `pydantic`, `pydantic-settings` | `Plan`/`TaskModel`/`TestResult` models and `Settings` environment loading. |
| `anthropic`, `openai` | The hosted providers; the OpenAI client is also the transport for Ollama. |
| `httpx` | HTTP client underpinning both provider SDKs. |
| `click` | The `furrow` command-line interface. |
| `rich` | Terminal rendering. |
| `aiofiles` | Async file reads and writes. |
| `tenacity` | Retry with exponential backoff on transient provider errors. |

Earlier revisions depended on `structlog` and `python-dotenv`; both were unused and have been removed. Dotenv-style configuration still works, via `pydantic-settings`' built-in `.env` support.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
mypy furrow
```

### Test suite

The suite is fully offline. `tests/conftest.py` provides `FakeLLM`, a scripted `LLMClient` subclass that never opens a socket and routes responses by prompt substring (`"planning agent"`, `"worker agent"`, `"tester agent"`), so the planner, workers and tester can be driven independently. No API key is needed to run it, and an autouse fixture strips every `FURROW_*` variable and the raw provider-key variables before each test so a developer's environment cannot change a result.

The same file installs a `sys.path` guard. A shared virtualenv's editable install can resolve `furrow` to a different worktree, which would silently test foreign code; `tests/conftest.py` forces the repository root onto `sys.path[0]` and then asserts that the resolved `furrow/config.py` is inside this checkout, failing loudly at import time if it is not.

| Module | Covers |
| --- | --- |
| `tests/conftest.py` | Fixtures: `FakeLLM`, `make_settings`, `tmp_workspace`, environment isolation, the path guard. |
| `tests/test_config.py` | `Settings` defaults, `env_prefix` loading and the Pydantic models. |
| `tests/test_llm.py` | Provider routing, the transient-error retry policy, file helpers. |
| `tests/test_agents.py` | JSON extraction, planner/worker prompts, the tester's runner detection and exit-code authority. |
| `tests/test_orchestrator.py` | Termination rules, dependency waves, the concurrency cap, `on_event`, the final report. |
| `tests/test_cli.py` | Click wiring: help text, every `start` flag landing on `settings`, the `--model` override, interrupt and failure exit codes. |
| `tests/test_web.py` | `/`, `/health`, `strip_markup`, and the `/ws` lifecycle: streaming, model override, payload validation, disconnect cancellation. |

`pytest --collect-only -q` reported 186 tests at the time of writing; the number changes as modules are added, so read it from the command rather than this table. `asyncio_mode = "auto"` is set in `pyproject.toml`, so async tests need no marker.

### Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request across a Python 3.10 and 3.12 matrix. `ruff check .`, `ruff format --check .` and `pytest` are blocking. `mypy furrow` is present but carries `continue-on-error: true`, so type errors stay visible in the job log without failing the run; the workflow file documents removing that line once the library is mypy-clean.

## License

This repository does not currently include a LICENSE file, so no license terms are declared. Add one before publishing.
