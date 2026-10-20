# Furrow

Autonomous coding agent with an infinite parallel development loop.

Furrow plans a goal into parallel tasks, executes them with worker agents that
actually **edit your files**, runs your test suite, and repeats until the goal
is satisfied.

## Install

```bash
pip install furrow
```

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment. To use a local
Ollama server instead, set `FURROW_PROVIDER=ollama` (no API key required).

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow web
```

### `furrow start [GOAL]`

| Option | Description |
| --- | --- |
| `--model` | Override the default model used by the agents |
| `--provider` | `anthropic`, `openai` or `ollama` |
| `--planner-model` | Override the planning model |
| `--worker-model` | Override the worker model |
| `--tester-model` | Override the tester model |
| `--max-cycles` | Stop after N cycles. `0` means unlimited |
| `--workspace` | Root directory the agents may read and write |

If `GOAL` is omitted you are prompted for it.

### `furrow web`

| Option | Description |
| --- | --- |
| `--host` | Bind address (default `0.0.0.0`) |
| `--port` | Bind port (default `8000`) |

Opens a dashboard that streams live agent progress over a websocket.

## Configuration

Every setting can be supplied as a `FURROW_`-prefixed environment variable or
in a `.env` file:

| Variable | Default | Description |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | `anthropic`, `openai` or `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Planning model |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Worker model |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Tester model |
| `FURROW_ANTHROPIC_API_KEY` | - | Anthropic API key |
| `FURROW_OPENAI_API_KEY` | - | OpenAI API key |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Concurrent worker limit |
| `FURROW_MAX_CYCLES` | `0` | Cycle limit, `0` means unlimited |
| `FURROW_WORKSPACE` | current directory | Root the agents may modify |
| `FURROW_LOG_LEVEL` | `INFO` | Structured logging level |

`ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are also read directly from the
environment.

## How it works

Each cycle runs four stages:

1. **Plan** - the planner turns the goal into 1-5 independent tasks. If the
   response is not valid JSON it is retried once with a stricter instruction.
2. **Execute** - tasks run in parallel, honouring their declared dependencies
   and the `max_parallel_tasks` cap.
3. **Test** - the tester detects your test runner, runs it with `cwd` set to
   the workspace, and decides pass/fail from the **exit code**. The model is
   only used to summarise failures, so a failing suite can never be reported as
   passing.
4. **Repeat** - the loop continues with the original goal plus a focus on the
   current failures.

The loop stops when the goal is complete, when `--max-cycles` is reached, or
when three consecutive cycles make no progress.

### Worker tools

Workers are not chatty assistants that print code - they edit files through a
bounded tool loop (8 rounds max) with three tools:

- `read_file` - read a file relative to the workspace
- `write_file` - write a file relative to the workspace
- `list_files` - list files under a directory

Every path is resolved against the workspace root. Anything that escapes it -
via `..` or an absolute path - is refused, so a worker cannot touch the rest
of your machine.

### Providers

Anthropic, OpenAI and Ollama are supported. Transient failures (connection
errors, timeouts, rate limits, 5xx) are retried with exponential backoff;
client errors are not retried.

## Development

```bash
pip install -e ".[dev]"

pytest -q                  # test suite
ruff check .               # lint
ruff format --check .      # formatting
mypy furrow                # type check
```

The suite covers the config models, the provider layer, the planner, the
worker tools and their sandboxing, the tester, the orchestrator loop, the CLI
and the websocket API. No test touches the network or a live model.

## License

MIT
