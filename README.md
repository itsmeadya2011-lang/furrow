# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
furrow start "Add JWT auth to the API"
furrow start "Add JWT auth to the API" --max-cycles 5
furrow web
```

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment.

## Configuration

Furrow reads configuration from environment variables with the `FURROW_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `FURROW_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai`, `ollama` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model for planning |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model for workers |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model for testing |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Max concurrent workers |
| `FURROW_MAX_CYCLES` | `0` | Max cycles (0 = unlimited) |
| `FURROW_ANTHROPIC_API_KEY` | - | Anthropic API key |
| `FURROW_OPENAI_API_KEY` | - | OpenAI API key |
| `FURROW_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama base URL |

## Architecture

Furrow uses a multi-agent architecture with a central orchestrator:

```
User Goal ──> Orchestrator ──> PlannerAgent (decomposes into 1-5 tasks)
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                    ▼
          WorkerAgent (parallel)              WorkerAgent (parallel)
                    │                                    │
                    └─────────────────┬──────────────────┘
                                      ▼
                              TesterAgent (runs tests)
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                    ▼
              Tests passed? ──> Yes: continue loop    No: re-plan with fix goal
```

See `.kilo/FURROW.md` for detailed architecture documentation.