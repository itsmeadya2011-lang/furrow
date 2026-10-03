# Furrow

Furrow is an autonomous coding agent. You give it a goal; it plans a slice of
work into tasks, runs those tasks in parallel through LLM worker agents that
write files inside your workspace, runs your project's own test command, and
feeds any failures back into the next round of planning. It repeats until the
goal is met or a stopping threshold is reached.

## Install

```bash
pip install furrow
```

Requires Python 3.10 or newer.

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow web
```

`furrow start` prompts for a goal if you omit it, and takes `--model` to
override the configured model. Ctrl-C exits with status 130.

`furrow web` serves the browser UI on port 8000 and binds `0.0.0.0`, so it
listens on every interface rather than loopback only. The page opens a
WebSocket and receives `accepted`, then zero or more `progress` messages, then
exactly one terminal `complete` or `error`.

Set a provider key first: `FURROW_ANTHROPIC_API_KEY` or
`FURROW_OPENAI_API_KEY`. Furrow falls back to the bare `ANTHROPIC_API_KEY` and
`OPENAI_API_KEY` when the prefixed names are unset. Ollama needs no key.

## Configuration

Furrow reads environment variables prefixed `FURROW_` and also reads a `.env`
file in the working directory.

| Variable | Default | Meaning |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | `anthropic`, `openai`, or `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Model used when a role model is empty |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Planning role |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Worker role |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Tester role |
| `FURROW_ANTHROPIC_API_KEY` | unset | Falls back to `ANTHROPIC_API_KEY` |
| `FURROW_OPENAI_API_KEY` | unset | Falls back to `OPENAI_API_KEY` |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint, read when the provider is `ollama` |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Ceiling on concurrent worker agents |
| `FURROW_MAX_CYCLES` | `0` | Cycle budget; `0` means unlimited |
| `FURROW_WORKSPACE` | current directory | Root for worker writes and test runs |
| `FURROW_LOG_LEVEL` | `INFO` | Accepted by the settings model, not read by any code today |
| `FURROW_WEB_HOST` | `0.0.0.0` | Bind address for `furrow web` |
| `FURROW_WEB_PORT` | `8000` | Port for `furrow web` |

## How it works

Each cycle does four things.

1. **Plan.** The planner returns JSON tasks, each with an `id`, a
   `description`, `files`, and `dependencies`.
2. **Execute.** Tasks run in dependency waves. Tasks whose dependencies are all
   satisfied run concurrently, up to `FURROW_MAX_PARALLEL_TASKS` at a time. A
   task whose dependency failed is marked `skipped`. Dependencies that cannot
   be resolved fall back to unordered execution instead of stalling.
3. **Test.** The tester runs the first available test runner in the workspace
   (`pytest`, `python -m pytest`, `npm test`, `pnpm test`, `yarn test`,
   `cargo test`, `go test`) under a 120-second timeout, then asks the model to
   read that output and return a JSON verdict.
4. **Repeat.** Failure summaries are prepended to the next planning prompt and
   cleared once a run is green. The goal string you passed in is never
   overwritten.

The loop stops when:

- a non-empty plan had every task completed and the tester reported `passed`, or
- any task failed, or
- 3 consecutive cycles raised an exception, or
- 3 consecutive cycles completed no task, or
- `FURROW_MAX_CYCLES` is reached, when set above `0`.

The last three are halts rather than successes, so a run can finish with the
goal unmet.

## Safety

- Worker agents write files and do nothing else. Furrow does not execute shell
  commands authored by a model.
- Worker writes are confined to the workspace. Absolute paths, `..` segments,
  and paths that resolve outside the workspace, including through symlinks,
  are rejected and reported in the task result.
- The tester runs the project's own test command with a 120-second timeout, and
  kills and reaps the child process if it overruns.

## Architecture

`.kilo/FURROW.md` covers the components, the termination and safety rules, and
the second implementation of this design under `.kilo/agent/`.