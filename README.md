# Furrow

Autonomous coding agent: it turns a goal into a task plan, works the tasks in
parallel dependency-ordered layers, runs your test suite, and repeats until the
tests are green. The loop is bounded, and always reports why it stopped.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow --version
furrow start "Add JWT auth to the API"
furrow web
```

`furrow start` prompts for a goal on stdin when you do not pass one.

### `furrow start`

| Flag | Default | Meaning |
| --- | --- | --- |
| `GOAL` | prompted | Objective for the run. |
| `--provider` | `FURROW_PROVIDER` (`anthropic`) | `anthropic`, `openai` or `ollama`. |
| `--model MODEL_ID` | per-role defaults | Sets the model for **every** role: planner, workers and tester. |
| `--max-parallel-tasks N` | `5` | Workers allowed to run at once (`>= 1`). Throttles concurrency; never drops planned work. |
| `--max-cycles N` | `0` | Stop after N cycles (`>= 0`). `0` means unlimited. |
| `--workspace DIR` | current directory | Directory the test suite is run in. |

Flags override the environment; credentials, provider defaults and timeouts come
from the environment or a `.env` file.

### `furrow web`

```bash
furrow web --host 127.0.0.1 --port 8000
```

Serves a one-page monitoring UI plus a `GET /health` endpoint returning
`{"status": "ok"}`. The page streams orchestrator events live over `WS /ws`: cycle
starts, plans, task outcomes, test verdicts, and a final `done` event naming the halt
reason. One run executes at a time per process; a second connection is refused, and
closing the page stops the run.

### How a run ends

Every exit emits a `done` event with one of these reasons:

- `goal complete` - tasks exist, every one is `COMPLETED`, none `FAILED`, tests passed.
- `max_cycles reached` - the `--max-cycles` / `FURROW_MAX_CYCLES` cap was hit.
- `no progress` - three consecutive cycles after a baseline cycle changed neither the
  completed/failed counts nor the test outcome; a fully stalled run halts on cycle 4.
- `no tasks planned` - the planner returned an empty plan.
- `error` - planning, a task, or the cycle itself failed.

## Configuration

Every field is read from the environment or from a `.env` file in the current
directory, with the `FURROW_` prefix.

| Setting | Default | Meaning |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | `anthropic`, `openai` or `ollama`. |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Fallback model id for `LLMClient.complete()` when a caller requests none. |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the planner. |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the workers. |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the tester. |
| `FURROW_ANTHROPIC_API_KEY` or `ANTHROPIC_API_KEY` | unset | Anthropic credential. Both spellings are accepted, from the environment or from `.env`. |
| `FURROW_OPENAI_API_KEY` or `OPENAI_API_KEY` | unset | OpenAI credential. Both spellings are accepted. |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL of the local Ollama server. |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Maximum concurrent worker tasks (`>= 1`). |
| `FURROW_MAX_CYCLES` | `0` | Cycle cap (`>= 0`); `0` means unlimited. |
| `FURROW_WORKSPACE` | current directory | Directory the test suite is run in. |
| `FURROW_REQUEST_TIMEOUT` | `120.0` | Per-request timeout, in seconds, for every provider call (`> 0`). |
| `FURROW_MAX_TOKENS` | `4096` | Output token budget per request (`>= 1`). |
| `FURROW_LOG_LEVEL` | `INFO` | Declared, but not yet applied to any logger. |

Selecting `FURROW_PROVIDER=openai` or `ollama` also swaps the default model ids to
that provider's (`gpt-4o` / `gpt-4o-mini`, `qwen2.5-coder` / `qwen2.5`) instead of
sending `claude-*` ids to a provider that has never heard of them. A model id you set
yourself is never overwritten.

## Security

`furrow web` is **unauthenticated**: no credentials, no origin check, no rate limiting.
Anyone who can reach it can start a run that spends provider API credits and runs your
test commands inside `FURROW_WORKSPACE`. `--host` defaults to `127.0.0.1`; keep it on loopback.

## Limits

- Workers are LLM-driven: `WorkerAgent` returns prose only, furrow does not apply file
  edits itself. Only the tester runs commands - the first available of `pytest`,
  `python -m pytest`, `npm test`, `pnpm test`, `yarn test`, `cargo test`, `go test`
  (60 s per runner, 90 s per sweep), classified by an LLM.
- `max_cycles=0` (unlimited) is a money-burning hot loop - bound it with `--max-cycles`
  or `FURROW_MAX_CYCLES`.

## Exit codes

`0` finished normally, `1` configuration or startup failure, `2` usage error, `130` Ctrl-C.

## Library use

`LLMClient()` with no arguments falls back to the global settings; every request is
bounded by `FURROW_REQUEST_TIMEOUT` and a truncated reply raises `LLMResponseError`.
Close with `async with LLMClient() as client:` or an idempotent `await client.aclose()`.
