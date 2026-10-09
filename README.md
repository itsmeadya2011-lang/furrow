# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow start "Add JWT auth" --provider ollama --max-cycles 10 --max-parallel 3
furrow state        # show persisted state from the last run
furrow reset        # clear persisted state
furrow web          # launch the web UI (http://localhost:8000)
```

State is persisted to `.furrow/state.json` in the workspace. The loop halts when tests pass, when no tasks remain, when `--max-cycles` is reached, or on Ctrl+C.

Providers: `anthropic` (default), `openai`, `ollama`. Set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in your environment, or use `FURROW_*` settings (e.g., `FURROW_PROVIDER`, `FURROW_MAX_CYCLES`).