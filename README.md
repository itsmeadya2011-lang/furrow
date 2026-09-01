# Furrow

Autonomous coding agent with an infinite parallel development loop. Furrow decomposes high-level goals into independent tasks, dispatches them to parallel worker agents, and validates results with a dedicated tester — all orchestrated in a continuous cycle until the objective is met.

## Features

- **Parallel development loop** — runs an autonomous plan → execute → test cycle that iterates until the goal is reached or a cycle limit is hit.
- **Planner agent** — analyzes the goal and current workspace state, then produces a structured plan of independent, parallelizable tasks.
- **Worker agents** — execute tasks concurrently, each in isolation, applying code changes, running commands, and reporting outcomes.
- **Tester agent** — validates worker output against acceptance criteria, running tests and static checks before marking a task complete.
- **Orchestrator** — coordinates the planner, workers, and tester, managing concurrency, retries, and cycle progression.
- **CLI and web interfaces** — drive Furrow from the terminal or monitor progress through a browser-based dashboard.

## Installation

Requires Python 3.10+.

```bash
pip install furrow
```

For development:

```bash
git clone https://github.com/example/furrow.git
cd furrow
pip install -e ".[dev]"
```

## Configuration

Furrow is configured via environment variables:

| Variable | Description |
| --- | --- |
| `ANTHROPIC_API_KEY` | API key for Anthropic (Claude) models. |
| `OPENAI_API_KEY` | API key for OpenAI (GPT) models. |
| `FURROW_PROVIDER` | Provider to use: `anthropic` or `openai`. |
| `FURROW_MODEL` | Model name (e.g. `claude-sonnet-4-20250514`, `gpt-4o`). |
| `FURROW_MAX_PARALLEL_TASKS` | Maximum number of worker tasks to run in parallel. |
| `FURROW_MAX_CYCLES` | Maximum number of development cycles before stopping. |
| `FURROW_LOG_LEVEL` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

Set at least one API key and a provider before running Furrow.

## Usage

### CLI

Start an autonomous development loop from the terminal:

```bash
furrow start "Add JWT auth to the API"
```

### Web

Launch the web dashboard to monitor and control Furrow:

```bash
furrow web
```

## Architecture

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│  Planner   │────▶│  Workers   │────▶│   Tester   │
│   agent    │     │ (parallel) │     │   agent    │
└────────────┘     └────────────┘     └────────────┘
       ▲                                    │
       └──────────── Orchestrator ◀─────────┘
```

1. **Planner** — receives the goal and current workspace, emits a plan of tasks.
2. **Orchestrator** — dispatches each task to a **Worker** up to `FURROW_MAX_PARALLEL_TASKS` at a time.
3. **Worker** — performs the task (code edits, commands, file operations) and returns an outcome.
4. **Tester** — verifies the outcome against acceptance criteria; on success the task is marked done, on failure it is re-queued.
5. The **Orchestrator** advances to the next cycle, invoking the Planner again with updated state, until the goal is satisfied or `FURROW_MAX_CYCLES` is reached.

## License

Furrow is open source. Licensed under the MIT License (see `LICENSE` file if provided).
