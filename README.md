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
furrow resume              # resume from .furrow_state.json
furrow resume --state-file /path/to/state.json  # resume from custom file
```

`furrow start` saves state to `.furrow_state.json`, which can be used with `resume` to continue a previous session.

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.
