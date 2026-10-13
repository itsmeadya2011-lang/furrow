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

Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or use Ollama with `FURROW_PROVIDER=ollama` in your environment.

## Features

- **Infinite development loop** - Plans, executes, tests, and repeats until the goal is complete
- **Parallel execution** - Runs up to 5 tasks simultaneously (configurable)
- **Persistent state** - Resumes from where it left off across sessions
- **Multiple LLM providers** - Anthropic, OpenAI, and Ollama support
- **Web UI** - Monitor progress at `http://localhost:8000` with `furrow web`
- **Mid-loop input** - Add new requirements while the loop is running

## Configuration

Environment variables (prefix with `FURROW_`):
- `PROVIDER` - `anthropic`, `openai`, or `ollama` (default: anthropic)
- `MODEL` - Default model for all agents
- `PLANNER_MODEL` - Model for planning (default: claude-3-5-haiku)
- `WORKER_MODEL` - Model for workers (default: claude-3-5-sonnet)
- `TESTER_MODEL` - Model for testing (default: claude-3-5-sonnet)
- `MAX_PARALLEL_TASKS` - Max concurrent workers (default: 5)
- `MAX_CYCLES` - Max development cycles (0 = unlimited)
- `OLLAMA_BASE_URL` - Ollama endpoint (default: http://localhost:11434)

## Architecture

Furrow uses a 4-agent architecture:
1. **Orchestrator** - Runs the infinite loop, manages state
2. **Planner** - Breaks goals into parallelizable tasks
3. **Workers** - Execute tasks with file read/write operations
4. **Tester** - Runs test suites and parses results directly

See `.kilo/FURROW.md` for details.
