# Furrow Agent Guidance

## Running Tests
```bash
python -m pytest tests/ -v
```

## Code Style
- Ruff for linting (line-length: 100)
- mypy for type checking
- Python >= 3.10

## Architecture
- `furrow/core/orchestrator.py` - Main loop controller
- `furrow/agents/planner.py` - Breaks goals into tasks
- `furrow/agents/worker.py` - Executes individual tasks
- `furrow/agents/tester.py` - Runs tests and reports results
- `furrow/llm.py` - LLM client (Anthropic/OpenAI)
- `furrow/config.py` - Pydantic models and settings
- `furrow/web/server.py` - FastAPI web UI
- `furrow/cli/main.py` - CLI entry point

## Key Patterns
- All async operations use `asyncio`
- LLM calls wrapped with tenacity retry
- Max parallel tasks controlled by `max_parallel_tasks` setting
- Max cycles enforced via `max_cycles` setting
- Orchestrator tracks plan state across cycles
