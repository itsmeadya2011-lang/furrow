# Furrow

Autonomous coding agent with a parallel development loop.

Furrow breaks a goal into independent tasks, executes the workers in parallel,
runs the project's test suite, and — if tests fail — loops again with a
"fix the failing tests" goal until everything passes (or an optional cycle limit
is reached).

## Install

```bash
pip install furrow
```

## Usage

```bash
# Run the autonomous loop from the command line
furrow start "Add JWT auth to the API"

# You can also be prompted interactively
furrow start

# Override the default model
furrow start "Add JWT auth to the API" --model claude-opus-4-20250514

# Launch the web UI (serves a small page + WebSocket-driven loop)
furrow web
```

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment (see Providers).

## Providers

Furrow supports three providers, selected with the `FURROW_PROVIDER` setting (or the
`provider` field in code):

| Provider   | Env vars                                                            |
| ---------- | ------------------------------------------------------------------- |
| `anthropic`| `ANTHROPIC_API_KEY` (default)                                       |
| `openai`   | `OPENAI_API_KEY`                                                    |
| `ollama`   | `FURROW_PROVIDER=ollama`, optional `FURROW_OLLAMA_BASE_URL`         |

Ollama is used through its OpenAI-compatible API (default base URL
`http://localhost:11434`).

## Configuration

All settings are overridable via environment variables prefixed with `FURROW_`.

| Setting                | Env var                      | Default                              | Purpose                                            |
| ---------------------- | ---------------------------- | ------------------------------------ | -------------------------------------------------- |
| `provider`             | `FURROW_PROVIDER`            | `anthropic`                          | Which LLM backend to use                           |
| `model`                | `FURROW_MODEL`               | `claude-sonnet-4-20250514`           | Default model for plain completions                |
| `planner_model`        | `FURROW_PLANNER_MODEL`       | `claude-3-5-haiku-20241022`          | Model used to plan tasks                           |
| `worker_model`         | `FURROW_WORKER_MODEL`        | `claude-3-5-sonnet-20241022`         | Model used to implement tasks                      |
| `tester_model`         | `FURROW_TESTER_MODEL`        | `claude-3-5-sonnet-20241022`         | Model used to evaluate test output                 |
| `max_parallel_tasks`   | `FURROW_MAX_PARALLEL_TASKS`  | `5`                                  | Max worker tasks executed concurrently             |
| `max_cycles`           | `FURROW_MAX_CYCLES`          | `0` (capped at a safety limit of 50) | Stop after N cycles; `0` = run until done          |
| `max_tokens`           | `FURROW_MAX_TOKENS`          | `4096`                               | Max tokens per completion                          |
| `retry_attempts`       | `FURROW_RETRY_ATTEMPTS`      | `3`                                  | Retries on transient API errors (exp. backoff)      |
| `ollama_base_url`      | `FURROW_OLLAMA_BASE_URL`     | `http://localhost:11434`             | Ollama endpoint                                    |

## How the loop works

1. **Plan** — the planner splits the goal into 1–5 independent, parallelizable tasks.
2. **Work** — workers run concurrently (bounded by `max_parallel_tasks`). A worker may
   write or overwrite files by including them in a fenced `edits` block after its summary:
   ````markdown
   ```edits
   [{"path": "src/foo.py", "content": "print('hello')\n"}]
   ```
   ````
3. **Test** — the tester runs the project's test suite (`pytest`, `npm test`, `cargo test`,
   `go test`, …) and asks the model whether they passed.
4. **Decide** — if tests pass (or the planner returns no tasks), Furrow halts. If they
   fail, the loop continues with a "fix the failing tests" goal, up to `max_cycles`.
