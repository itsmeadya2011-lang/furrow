# Furrow

Autonomous coding agent with a bounded development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow web
```

The loop is bounded by `max_cycles` (default: 25; 0 = unlimited). Web UI defaults to localhost; set `FURROW_WEB_TOKEN` or `web_token` to enable auth. Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.
