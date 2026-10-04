# Furrow

Autonomous coding agent with an infinite parallel development loop.

Every cycle plans a set of parallel tasks, runs them concurrently, then runs the
tests. The loop only halts when the plan is non-empty, every task completed, and
the tests passed; anything else and it plans another cycle.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API" --model claude-sonnet-4-20250514 --cycles 5 --max-parallel 3
furrow web --host 0.0.0.0 --port 8000 --cycles 5
```

`furrow web` streams the run to the browser over a websocket: live events, the
current cycle number, and a Stop button that cancels the running agent.

## Configuration

Pick a provider with `FURROW_PROVIDER`: `anthropic` (default), `openai`, or `ollama`.

- `anthropic` / `openai` need `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, or the
  prefixed `FURROW_ANTHROPIC_API_KEY` / `FURROW_OPENAI_API_KEY`.
- `ollama` needs no API key. `FURROW_OLLAMA_BASE_URL` (default
  `http://localhost:11434`) is reached through its OpenAI-compatible `/v1` endpoint.

The loop is unbounded by default, so `FURROW_MAX_CYCLES` is the runaway guard:
`0` means unbounded, any other value stops the run after that many cycles
(`--cycles N` does the same for a single invocation). `FURROW_MAX_PARALLEL_TASKS`
(default `5`, or `--max-parallel N`) caps concurrent workers per cycle.