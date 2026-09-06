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

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FURROW_PROVIDER` | LLM provider: `anthropic`, `openai`, or `ollama` |
| `FURROW_MODEL` | Default model override |
| `FURROW_PLANNER_MODEL` | Model for planning |
| `FURROW_WORKER_MODEL` | Model for workers |
| `FURROW_TESTER_MODEL` | Model for testing |
| `FURROW_MAX_CYCLES` | Max development cycles (0 = unlimited) |
| `FURROW_JOURNAL_PATH` | Path to append JSONL journal entries |
| `FURROW_LOG_LEVEL` | Logging level |

## Architecture

Furrow runs an infinite loop:
1. **Plan** - Break goal into parallelizable tasks
2. **Execute** - Spawn worker subagents (respecting dependencies)
3. **Test** - Verify with linters and test suites
4. **Repeat** - Pass results forward for context-aware replanning

Workers read files before editing, run quick verification, and return structured changes.
The tester parses output robustly across multiple test runners.
