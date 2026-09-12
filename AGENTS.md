# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project: Furrow

Furrow is an autonomous coding agent that runs an infinite parallel development loop.

## Architecture

```
furrow/
  __init__.py        # Public API: LLMClient, Settings
  config.py           # Models: Provider, TaskModel, Plan, TestResult, Settings
  llm.py              # LLMClient: Anthropic, OpenAI, Ollama providers
  core/
    __init__.py       # Orchestrator export
    orchestrator.py   # Infinite loop: plan -> execute -> test -> repeat
  agents/
    __init__.py       # PlannerAgent, WorkerAgent, TesterAgent exports
    planner.py        # Breaks goals into parallelizable tasks (JSON output)
    worker.py         # Executes a single task via LLM
    tester.py         # Runs tests, parses results, reports pass/fail
    prompts.py        # System prompts for planner, worker, tester
  cli/
    main.py           # CLI: `furrow start`, `furrow web`
  web/
    __init__.py       # FastAPI app and run() export
    server.py         # Web UI with WebSocket streaming
```

## Key Design Decisions

- **Orchestrator loop**: Stores tasks as instance state (`self.tasks`) so `_is_done()` correctly tracks progress across cycles
- **Max cycles**: Configurable via `Settings.max_cycles` (0 = unlimited)
- **Max parallel**: Configurable via `Settings.max_parallel_tasks`, enforced with `asyncio.Semaphore`
- **LLM providers**: Supports Anthropic, OpenAI, and Ollama (OpenAI-compatible)
- **Logging**: Uses `structlog` for structured logging
- **Retry**: LLM calls use `tenacity` with exponential backoff on ConnectionError/TimeoutError
- **Graceful shutdown**: `Orchestrator.request_stop()` sets a flag checked at the top of each cycle

## Running Tests

```bash
pytest tests/ -v
```

## Running the CLI

```bash
furrow start "Your goal here"
furrow web
```

## Configuration

Environment variables prefixed with `FURROW_` or `.env` file:
- `FURROW_PROVIDER`: anthropic | openai | ollama
- `FURROW_MODEL`: LLM model ID
- `FURROW_MAX_CYCLES`: Limit on cycles (0 = unlimited)
- `FURROW_MAX_PARALLEL_TASKS`: Max concurrent worker tasks
- `FURROW_ANTHROPIC_API_KEY` / `FURROW_OPENAI_API_KEY`
- `FURROW_OLLAMA_BASE_URL`: Ollama server URL
