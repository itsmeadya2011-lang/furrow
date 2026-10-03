# Furrow

Autonomous coding agent with a plan/execute/test loop. Given a goal, Furrow asks a
planner to break it into 1-5 independent tasks, runs one worker per task in
parallel, runs your project's test command, and starts the next cycle. It stops
when the planner reports the goal complete, when the cycle cap is reached, or
when you stop it.

## Install

```bash
pip install furrow          # from a release
pip install -e ".[dev]"     # from a checkout; adds pytest, ruff, mypy
```

Requires Python 3.10 or newer.

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow web
```

### `furrow start [GOAL]`

Runs the loop. `GOAL` is optional: a bare `furrow start` prompts for one
interactively.

| Option | Description |
| --- | --- |
| `--model TEXT` | Override the default LLM model. |
| `--provider [anthropic\|openai\|ollama]` | LLM provider to use. |
| `--max-cycles INTEGER` | Maximum plan/execute/test cycles; `0` = unlimited. |
| `--max-parallel-tasks INTEGER` | Maximum number of tasks to run in parallel. |
| `--workspace PATH` | Working directory for the run (the test command's `cwd`). |
| `--dry-run` | Resolve configuration, print it, and exit without running. |
| `--planner-model TEXT` | Override the planner LLM model. |
| `--worker-model TEXT` | Override the worker LLM model. |
| `--tester-model TEXT` | Override the tester LLM model. |

`--dry-run` calls no LLM at all — it prints the resolved provider, models,
workspace, cycle cap, parallelism, and test settings, then exits.

On a normal halt the CLI prints a summary panel: halt reason, cycle count, task
tallies, the last test result, and a per-cycle report table. Ctrl+C aborts the
run instead, printing `Furrow stopped by user.` and exiting 0.

### `furrow web`

Serves a single-page console (`--host`, default `127.0.0.1`; `--port`, default
`8000`). Enter a goal in the page and the same loop runs server-side, streaming
progress lines to the browser over a websocket at `/ws`. Closing the socket stops
the run.

## API keys

The selected provider needs credentials before it will run:

| Provider | Requirement |
| --- | --- |
| `anthropic` (default) | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `ollama` | None — POSTs to `{FURROW_OLLAMA_BASE_URL}/api/chat` |

## Configuration

`Settings` (`furrow/config.py`) is a `pydantic-settings` model configured with
`env_prefix="FURROW_"` and `env_file=".env"`, so every field is settable as
`FURROW_<FIELD>` in the environment or in a `.env` file. Command-line flags take
precedence over environment values.

| `FURROW_*` | Default | Effect |
| --- | --- | --- |
| `provider` | `anthropic` | `anthropic`, `openai`, or `ollama`. |
| `model` | `claude-sonnet-4-20250514` | Default model for completions. |
| `planner_model` | `claude-3-5-haiku-20241022` | Model used by the planner. |
| `worker_model` | `claude-3-5-sonnet-20241022` | Model used by workers. |
| `tester_model` | `claude-3-5-sonnet-20241022` | Model used by the tester. |
| `anthropic_api_key` | unset | Anthropic key; falls back to `ANTHROPIC_API_KEY`. |
| `openai_api_key` | unset | OpenAI key; falls back to `OPENAI_API_KEY`. |
| `ollama_base_url` | `http://localhost:11434` | Ollama server root. |
| `ollama_model` | `llama3` | Ollama model; falls back to `model`. |
| `max_parallel_tasks` | `5` | Worker concurrency cap per cycle. |
| `max_cycles` | `0` | Cycle limit for the loop; `0` = unlimited. |
| `max_task_retries` | `1` | Extra attempts for a task that raises. |
| `test_command` | unset | Explicit test command; when unset it is auto-detected. |
| `test_timeout` | `120` | Seconds before the test process is killed (explicit `test_command` only). |
| `llm_max_retries` | `3` | Retries for transient provider errors (429/5xx/timeouts). |
| `llm_max_tokens` | `4096` | `max_tokens` for Anthropic completions. |
| `workspace` | current directory | Working directory; used as the test command's `cwd`. |
| `log_level` | `INFO` | Root `logging` level applied on import. |
| `web_host` | `127.0.0.1` | Bind address for `furrow web`. |
| `web_port` | `8000` | Bind port for `furrow web`. |

## The loop

Each cycle:

1. **Plan.** The planner is given the original goal, plus the previous cycle's
   test failures as extra context when there were any. The goal itself is never
   rewritten. It returns a JSON task list, or an empty one meaning "done".
2. **Execute.** Each task gets its own worker, run concurrently up to
   `max_parallel_tasks`. Tasks are de-duplicated by `id` across cycles; a task
   that already completed or was skipped is not re-run.
3. **Test.** The tester runs `test_command`, or, when that is unset, the first
   available of `pytest -q`, `python -m pytest -q`, `npm test`,
   `pnpm test`, `yarn test`, `cargo test -q`, `go test ./...`. The process exit
   code is the authoritative pass/fail signal; the model only writes the summary
   and the failure list.

The loop halts when:

- **The planner returns an empty task list** — the goal is complete.
- **`max_cycles` is reached** — `0` means unlimited.
- **`stop()` is called** — Ctrl+C in the CLI, or the browser websocket closing.
- **The planner or tester raises** — recorded as `stopped_reason="error"` in the
  run result rather than propagating the exception.

A red test suite keeps the loop alive so the next cycle can plan fixes, with one
guard: that only applies if the previous cycle actually planned new runnable
work. Otherwise, once every tracked task has settled and none failed, Furrow
treats the goal as met. Planner output and worker summaries are recorded in the
in-memory run result; nothing is persisted to disk.

## Development

```bash
pytest                 # test suite
ruff check             # lint
mypy furrow            # type check
```

Tooling configuration lives in `pyproject.toml`: `[tool.pytest.ini_options]`
(`asyncio_mode = "auto"`, `testpaths = ["tests"]`), `[tool.ruff]`
(line length 100, `py310`, rules `E,F,I,UP,B,W`), and `[tool.mypy]`.

## Not implemented

Worth knowing before you rely on the loop:

- **The worker agent cannot edit files.** `WORKER_PROMPT` states it has no tool
  access; it returns a text implementation summary for a follow-up actor to apply.
- **No state persists across sessions.** Goal, tasks, and test results live in a
  `RunResult` in memory for the duration of one run.
- **The web UI is one page**, not a dashboard: a goal box and a log stream.
