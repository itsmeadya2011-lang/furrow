# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow start "Fix flaky tests" --max-cycles 5 --model gpt-4o
furrow web
```

## Providers

Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or point `FURROW_OLLAMA_BASE_URL` at a
local Ollama instance. Switch providers via `FURROW_PROVIDER=anthropic|openai|ollama`.
