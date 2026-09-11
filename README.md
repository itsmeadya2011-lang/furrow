# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow start "Add JWT auth" --max-cycles 5
furrow run
furrow doctor
furrow web
```

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.

## Commands

| Command | Description |
|---------|-------------|
| `furrow start [goal]` | Start the development loop with a goal |
| `furrow run` | Start interactively (prompt for goal) |
| `furrow doctor` | Run environment diagnostics |
| `furrow web` | Start the web UI |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | from config | Override LLM model |
| `--max-cycles` | 0 (unlimited) | Max cycles before auto-halt |
