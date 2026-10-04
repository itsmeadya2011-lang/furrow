# Furrow

Autonomous coding agent that loops **plan -> execute -> test** until the goal is met or a
limit is reached. Each cycle plans 1-5 tasks, runs them in dependency-ordered parallel waves,
runs the project's test suite, and feeds any failures back into the next cycle's plan.

The loop is bounded by `max_cycles` (`0`, the default, means unlimited). It also halts when
every planned task completed and the tests passed, or after `fail_fast_cycles` cycles in a
row made no progress -- each either raised an unexpected exception or finished with a failing
test run. A cycle whose tests pass, or that planned no tasks, clears that streak, so a
persistently failing suite cannot cycle indefinitely under the unlimited default.

## Install

Not published to PyPI. Install from a source checkout:

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.10.

## Usage

```bash
furrow start "Add a /health endpoint"   # run the loop
furrow web                               # serve the web UI on 127.0.0.1:8000
furrow --version
```

`furrow start` prompts for a goal when the argument is empty.

### `furrow start [GOAL]`

| Flag | Meaning |
| --- | --- |
| `--provider {anthropic,openai,ollama}` | LLM provider. Default: `$FURROW_PROVIDER`, else `anthropic`. |
| `--model MODEL` | Sets the model for **every** role (planner, worker, tester). |
| `--workspace PATH` | Directory to work in. Must exist. Default: `$FURROW_WORKSPACE`, else cwd. |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Default: `$FURROW_LOG_LEVEL`, else `INFO`. |
| `-v`, `--verbose` | Shorthand for `--log-level DEBUG`. |
| `--max-cycles N` | Stop after N cycles (`0` = unlimited). Default: `$FURROW_MAX_CYCLES`, else `0`. |

### `furrow web`

Accepts the same `--provider` / `--model` / `--workspace` / `--log-level` / `-v` options,
plus:

| Flag | Meaning |
| --- | --- |
| `--host HOST` | Interface to bind. Default `127.0.0.1`. |
| `--port PORT` | Port to bind. Default `8000`. |
| `--reload` | Restart the server when the code changes. |

## Configuration

Every setting is a `FURROW_`-prefixed environment variable; values may also come from a
`.env` file in the current working directory. A malformed value is downgraded to a logged
warning and the default is used, rather than raising on import.

| Variable | Default | Meaning |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | `anthropic`, `openai`, or `ollama`. |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model id. |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Planner model id. |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Worker model id. |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Tester model id. |
| `FURROW_ANTHROPIC_API_KEY` | unset | Anthropic credential. |
| `FURROW_OPENAI_API_KEY` | unset | OpenAI credential. |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (unused; see Limitations). |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Concurrent workers per wave. |
| `FURROW_MAX_CYCLES` | `0` | Cycle ceiling; `0` = unlimited. |
| `FURROW_FAIL_FAST_CYCLES` | `3` | Consecutive non-progressing cycles (hard failure or failing test run) tolerated before halting. |
| `FURROW_MAX_TOOL_TURNS` | `10` | Tool-call rounds a worker gets per task. |
| `FURROW_TEST_TIMEOUT` | `120` | Seconds allowed for one test command run. |
| `FURROW_TEST_COMMAND` | unset | Explicit test command. When unset, a runner is detected. |
| `FURROW_WORKSPACE` | cwd | Root directory for all file access. |
| `FURROW_LOG_LEVEL` | `INFO` | Root log level. |

### API keys

The settings fields are `FURROW_ANTHROPIC_API_KEY` and `FURROW_OPENAI_API_KEY`. The bare
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` environment variables are also read, as a fallback,
when the corresponding `FURROW_`-prefixed field is unset.

```bash
export FURROW_ANTHROPIC_API_KEY=sk-ant-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

### Testing

When `FURROW_TEST_COMMAND` is unset, `TesterAgent` probes for a runner in this order and uses
the first one that is installed and actually runs: `pytest -q`, `python -m pytest -q`,
`npm test -- --silent`, `pnpm test -- --silent`, `yarn test --silent`, `cargo test -q`,
`go test ./...`. The process exit code is authoritative: a non-zero exit or a timeout is a
failure even if the model's verdict says otherwise.

## Development

```bash
pytest            # asyncio_mode = "auto", testpaths = ["tests"]
ruff check .
ruff format .     # line-length = 100, target-version = "py310"
mypy furrow       # [tool.mypy] in pyproject.toml
```

Ruff lint rules: `select = ["E", "F", "I", "UP", "B", "SIM"]`.

## Architecture

```
furrow/
  config.py              # Provider, TaskStatus, TaskModel, Plan, TestResult, Settings
  llm.py                 # LLMClient: complete(), agent_loop(), sandboxed file helpers
  core/orchestrator.py   # Orchestrator: the plan/execute/test loop
  agents/planner.py      # PlannerAgent -> Plan
  agents/worker.py       # WorkerAgent -> WorkerResult (real file tools)
  agents/tester.py       # TesterAgent -> TestResult (runs the test command)
  agents/prompts.py      # PLANNER_PROMPT, WORKER_PROMPT, TESTER_PROMPT
  cli/main.py            # `furrow start`, `furrow web`, `furrow --version`
  web/server.py          # FastAPI app: GET /, WebSocket /ws
```

Worker file tools (`read_file`, `write_file`, `edit_file`, `list_files`) are confined to
`settings.workspace`; paths that resolve outside it are refused.

`furrow/` is the standalone implementation. A parallel set of Kilo agent definitions in
`.kilo/agent/` plus the `/furrow` command drive the same loop through Kilo's subagent
tooling; see `.kilo/FURROW.md`.

## Limitations

- **`Provider.OLLAMA` is declared but not implemented.** The enum value, the
  `FURROW_OLLAMA_BASE_URL` setting, and the Ollama entry in the CLI's per-provider model
  defaults all exist, but `LLMClient.complete` and `LLMClient.agent_loop` raise a
  `ValueError` that names the supported providers. Use `anthropic` or `openai`.
- **The web server has no authentication or authorization.** It binds `127.0.0.1` by
  default; do not expose it publicly, and do not bind `0.0.0.0` on an untrusted network.
  Anyone who can reach it can start runs that write files in the workspace.
- **`mypy` is configured but was not run to a clean state.** `[tool.mypy]` enables
  `disallow_untyped_defs`, `disallow_incomplete_defs` and friends; treat mypy errors as
  outstanding work, not as a passing baseline.
- **Test suite status is not asserted here.** `tests/` is in progress.
- The worker edits real files. `settings.workspace` is a confinement boundary, not a
  sandbox with OS-level isolation; a misbehaving model can still damage anything inside it.