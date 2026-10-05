# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow web
```

`furrow start` takes optional overrides for the settings below. A flag that is not
passed leaves the configured value (environment, `.env`, or default) untouched.

| Flag | Description |
| --- | --- |
| `--model` | Override the LLM model. |
| `--planner-model` | Override the planner model. |
| `--worker-model` | Override the worker model. |
| `--tester-model` | Override the tester model. |
| `--provider` | Provider to use: `anthropic`, `openai` or `ollama`. Omit to keep the configured default. |
| `--max-cycles` | Stop after N planning cycles. `0` means unlimited. Must be 0 or greater. |
| `--max-parallel-tasks` | Maximum tasks to run in parallel per cycle. Must be 1 or greater. |

`furrow web` serves the UI on `/`. The same start endpoint also accepts an optional
`model` field, which sets all four model settings at once.

## Configuration

Configuration comes from environment variables and from a `.env` file in the
working directory. Every `furrow.config.Settings` field is read with the
`FURROW_` prefix.

| Variable | Default | Description |
| --- | --- | --- |
| `FURROW_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, or `ollama`. |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Model used by `LLMClient.complete` when no model is passed. Each agent passes its own model setting, so this is the fallback. |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model used by the planner agent. |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the worker agents. |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model used by the tester agent. |
| `FURROW_MAX_CYCLES` | `0` | Maximum plan/execute/test cycles. `0` means unlimited. |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Maximum tasks run concurrently in one cycle. |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL used when the provider is `ollama`. `/v1` is appended if it is missing, and an empty value falls back to this default. |
| `FURROW_WORKSPACE` | current working directory | Directory used as the `cwd` for test-runner subprocesses. If it is not a directory it is ignored (with a warning) and the process working directory is kept. |
| `FURROW_LOG_LEVEL` | `INFO` | Unused; declared on `Settings` but never read. |
| `FURROW_ANTHROPIC_API_KEY` | unset | Key for the `anthropic` provider. |
| `FURROW_OPENAI_API_KEY` | unset | Key for the `openai` provider. |
| `FURROW_LLM_MAX_RETRIES` | `3` | Extra attempts after the first one, so the default of `3` means up to `4` calls in total. Only transient failures are retried — HTTP 408, 409, 425, 429, 5xx and connection/timeout errors; 401/403, other 4xx and cancellation are raised immediately. A non-numeric or negative value falls back to `3`. |
| `FURROW_LLM_BACKOFF_SECONDS` | `1.0` | Base delay before a retry, doubled per attempt (`1.0` gives waits of 1s, 2s, 4s). A non-numeric value falls back to `1.0`. |
| `FURROW_TEST_TIMEOUT_SECONDS` | `120` | Timeout for a single test run. A non-numeric or non-positive value is logged and falls back to `120`. |

The LLM client also honours the unprefixed `ANTHROPIC_API_KEY` and
`OPENAI_API_KEY`; the `FURROW_`-prefixed settings win when both are set, and
`ValueError` is raised if neither is set for the active provider. The `ollama`
provider needs no key.

Example `.env`:

```dotenv
FURROW_PROVIDER=anthropic
FURROW_MAX_CYCLES=5
FURROW_MAX_PARALLEL_TASKS=8
ANTHROPIC_API_KEY=sk-ant-...
```
