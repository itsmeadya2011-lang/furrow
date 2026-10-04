# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

Furrow is not on PyPI. Install from a source checkout:

```bash
pip install -e ".[dev]"
```

The CLI entry point is `furrow.cli.main:main`. In a bare sandbox venv `PYTHONPATH` alone may not resolve the console script, so run with:

```bash
python -m furrow.cli.main <command>
```

## Usage

```bash
python -m furrow.cli.main start "Add JWT auth to the API"
python -m furrow.cli.main web
```

## Architecture

The orchestrator runs a planner -> worker -> tester loop until a stop condition is met. Each cycle:

1. **Plan** — the planner agent breaks the goal into 1–5 tasks with dependencies.
2. **Execute** — worker agents run concurrently (up to `max_parallel_tasks`, default 4) in dependency-respecting waves, writing real files into the workspace.
3. **Test** — the tester agent runs the project's test runner and judges the output.
4. **Repeat** — failures are fed back to the planner for the next cycle.

Progress events are emitted on an in-process event bus, which both the rich console and the web UI subscribe to.

## Stop Conditions

The loop is **not** unconditional. It stops when one of these becomes true:

- Tests pass.
- `max_cycles` is reached (`0` means unlimited).
- 3 consecutive cycles in which no worker wrote any file (`DEFAULT_NO_PROGRESS_LIMIT`).

## Settings

Environment variables use the `FURROW_` prefix. The bare forms of `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are also accepted for convenience.

| Variable | Default | Description |
|---|---|---|
| `FURROW_PROVIDER` | `anthropic` | LLM backend: `anthropic`, `openai`, or `ollama`. |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Fallback model for any role. |
| `FURROW_PLANNER_MODEL` | `claude-sonnet-4-20250514` | Model used by the planner. |
| `FURROW_WORKER_MODEL` | `claude-sonnet-4-20250514` | Model used by the worker. |
| `FURROW_TESTER_MODEL` | `claude-sonnet-4-20250514` | Model used by the tester. |
| `FURROW_ANTHROPIC_API_KEY` or `ANTHROPIC_API_KEY` | `None` | Anthropic credential. Excluded from logs. |
| `FURROW_OPENAI_API_KEY` or `OPENAI_API_KEY` | `None` | OpenAI credential. Excluded from logs. |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama daemon URL. |
| `FURROW_OLLAMA_MODEL` | `llama3.1` | Model id requested from Ollama. |
| `FURROW_WORKSPACE` | process CWD | Project root the agents operate on. |
| `FURROW_MAX_CYCLES` | `0` | Maximum planner cycles; `0` means unlimited. |
| `FURROW_MAX_PARALLEL_TASKS` | `4` | Maximum tasks executed concurrently per cycle. |
| `FURROW_MAX_TOKENS` | `8192` | Completion token budget requested from the provider. |
| `FURROW_TEST_COMMAND` | `None` | Explicit test command override, e.g. `"pytest -q"`. `None` means auto-detect. |
| `FURROW_TEST_TIMEOUT` | `120` | Per-run test timeout in seconds. |
| `FURROW_LOG_LEVEL` | `INFO` | Root log level name. |
| `FURROW_LOG_JSON` | `False` | Emit structlog JSON instead of console output. |

## CLI Surface

| Command | Flags | Description |
|---|---|---|
| `start [GOAL]` | `--model`, `--planner-model`, `--worker-model`, `--tester-model`, `--provider` (`anthropic`/`openai`/`ollama`), `--workspace`, `--max-cycles`, `--max-parallel-tasks`, `--test-command`, `--test-timeout`, `--log-level`, `--dry-run`, `--json` | Run the development loop. |
| `web` | `--host` (default `127.0.0.1`), `--port` (default `8000`) | Serve the web monitoring UI. |

Exit codes: `0` success, `1` runtime failure, `2` usage error, `130` interrupted.

## Security

- The web UI is **unauthenticated** and drives an agent with file-write and subprocess access. It binds `127.0.0.1` by default and warns loudly on any other host.
- The worker rejects absolute paths, `..` traversal and symlink escapes, so a model cannot write outside the workspace.

## Development

```bash
pytest tests
ruff check
mypy furrow
```

The suite currently runs **263 tests** across 8 files:

- `tests/test_cli.py`
- `tests/test_config.py`
- `tests/test_core.py`
- `tests/test_jsonutil.py`
- `tests/test_models.py`
- `tests/test_orchestrator.py`
- `tests/test_web.py`

`tests/test_llm.py` and `tests/test_agents.py` do not exist yet (they are being written by other agents).
