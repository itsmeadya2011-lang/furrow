# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API"     # run the loop
furrow start --max-cycles 3 --parallel 4 "Ship login page"
furrow status                              # show saved loop state
furrow web                                 # web UI with live event stream
```

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.
