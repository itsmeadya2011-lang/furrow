# Furrow

Autonomous coding agent with an infinite parallel development loop.

Furrow takes a high-level goal (e.g. *"Add JWT auth to the API"*) and runs an
iterative loop: it **plans** the work into sub-tasks, **executes** those tasks
in parallel across multiple worker agents, and **tests** the results after each
cycle. The loop repeats until convergence or until `max_cycles` is reached,
building on prior progress so that large features converge autonomously.

## Install

```bash
pip install furrow
```

For development, install from the repository with the `dev` extra:

```bash
git clone https://github.com/your-org/furrow
cd furrow
pip install -e .[dev]
```

## Configuration

All settings live in `furrow/config.py` (`Settings`) and can be provided via
environment variables prefixed with `FURROW_`, or in a `.env` file at the
project root. Copy [.env.example](.env.example) to `.env` and edit as needed.

| Variable | Default | Description |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, or `ollama`. |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default generation model. |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the planner to break goals into tasks. |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by workers to implement tasks. |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the tester to verify results. |
| `FURROW_ANTHROPIC_API_KEY` | — | Anthropic API key (required when `provider=anthropic`). |
| `FURROW_OPENAI_API_KEY` | — | OpenAI API key (required when `provider=openai`). |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (used when `provider=ollama`). |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Max parallel worker tasks per cycle. |
| `FURROW_MAX_CYCLES` | `0` | Max cycles to run; `0` = infinite. |
| `FURROW_WORKSPACE` | `.` | Directory to operate in. |
| `FURROW_LOG_LEVEL` | `INFO` | Logging verbosity. |

## Usage

Run a single development loop from the command line:

```bash
furrow start "Add JWT auth to the API"
```

Or start the web server to monitor and control runs in a browser:

```bash
furrow web
```

Set `FURROW_ANTHROPIC_API_KEY` or `FURROW_OPENAI_API_KEY` in your environment (or in
a `.env` file) before running.

### Kilo integration

When running inside [Kilo](https://kilo.ai/docs), Furrow is available as a
built-in command. From a Kilo chat, invoke:

```
/furrow Add JWT auth to the API
```

This runs the same planning → parallel execution → testing loop as the CLI, with
output surfaced back into the Kilo session.

## Architecture

```
              ┌──────────┐
   goal ────► │ Planner  │  (breaks goal into a Plan of tasks)
              └────┬─────┘
                   │ tasks
                   ▼
              ┌──────────────┐
              │ Orchestrator │  (schedules N parallel Workers per cycle)
              └──────┬───────┘
                     │ cycle results
             ┌────────┴────────┐
             ▼                 ▼
        ┌─────────┐       ┌────────┐
        │ Workers │       │ Tester │ (runs tests, reports TestResult)
        └─────────┘       └────────┘
```

- **Planner** (`furrow/agents/planner.py`) — converts a goal into a `Plan` of
  `TaskModel`s with dependencies and file assignments.
- **Orchestrator** (`furrow/core/orchestrator.py`) — runs the loop: plan,
  dispatch independent tasks in parallel to workers, test, then repeat.
- **Worker** (`furrow/agents/worker.py`) — implements individual tasks against
  the configured LLM.
- **Tester** (`furrow/agents/tester.py`) — verifies changes and produces a
  `TestResult`.

Per-cycle progress (tasks, results, status) is persisted under
`.furrow/state.json`, so an interrupted run can be resumed.

## Contributing

Install the development dependencies (see the `dev` extra above), then run:

```bash
# Tests
pytest

# Lint
ruff check .

# Type check
mypy furrow
```

## License

See [LICENSE](LICENSE).
