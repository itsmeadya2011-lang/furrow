# Furrow

Autonomous coding agent with an infinite parallel development loop. Furrow breaks a high-level goal into parallelizable tasks, executes them concurrently via worker subagents, tests the result, and repeats until the goal is complete.

## Architecture

Furrow is composed of five collaborating components:

| # | Component | Type | Role |
|---|-----------|------|------|
| 1 | **Orchestrator** (`furrow`) | Primary agent | Runs the infinite loop. Receives the goal, plans, delegates to workers, tests, reports, and repeats. |
| 2 | **Planner** (`furrow-planner`) | Subagent | Breaks the goal into 1–5 independent, parallelizable tasks. Outputs JSON with `tasks[]` (each with `id`, `description`, `files`, `dependencies`). |
| 3 | **Worker** (`furrow-worker`) | Subagent | Executes a single assigned task with minimal, targeted changes. Runs in 1–3 tool call rounds. No scope creep. |
| 4 | **Tester** (`furrow-tester`) | Subagent | Runs the project's test suite, linter, and type checker; fixes failures. Returns a `TestResult` JSON (`passed`, `summary`, `failures`). |
| 5 | **Command** (`/furrow`) | Slash command | Entry point. Invokes the `furrow` orchestrator with the goal from `$ARGUMENTS`. |

## Development Loop

Furrow runs an infinite cycle until the goal is complete or the user stops the session:

1. **Plan** — The planner breaks the current goal into 1–5 parallelizable tasks.
2. **Execute** — Workers run in parallel (up to `max_parallel_tasks`, default 5) to implement each task.
3. **Test** — The tester runs tests, lint, and type checks; failures are fed back as fixes in the next cycle.
4. **Repeat** — The orchestrator reports progress and immediately starts the next cycle with updated context.

Mid-loop user input is incorporated into the next cycle's plan. If tests fail, the next cycle's goal becomes "Fix failing tests…" so the loop converges.

## Install

```bash
pip install furrow
```

## Usage

Start the development loop from the Kilo TUI:

```bash
/furrow Add JWT authentication to the API with tests
```

Or from the CLI directly:

```bash
furrow start "Add JWT authentication to the API with tests"
furrow web   # run the web monitoring server
```

Set one of the following environment variables to authenticate with an LLM provider:

```bash
export ANTHROPIC_API_KEY=sk-...
# or
export OPENAI_API_KEY=sk-...
```

Settings can also be configured via `FURROW_*` environment variables or a `.env` file (e.g. `FURROW_PROVIDER`, `FURROW_MODEL`, `FURROW_MAX_CYCLES`).

## Configuration

Project configuration lives in `kilo.json`, which selects `furrow` as the default agent and grants open permissions (`bash`, `edit`, `read`) by default.

## Persistent State

Furrow tracks goal and task progress across sessions in a persistent state file at `.kilo/state/furrow-state.json`. Each cycle the orchestrator updates the file with the current goal, cycle number, completed and remaining tasks, worker results, and test outcomes, so a resumed session picks up where the last one left off.
