# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
# Start a new task
furrow start "Add JWT auth to the API"

# Resume from saved state
furrow start "Add JWT auth to the API" --resume

# Limit cycles
furrow start "Fix bug" --max-cycles 10

# Start web UI
furrow web
```

## Configuration

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment or `.env` file.

Create a `furrow.toml` in your project root:

```toml
provider = "anthropic"
model = "claude-sonnet-4-20250514"
max_parallel_tasks = 5
max_cycles = 0
workspace = "."
log_level = "INFO"
```

Or add to `pyproject.toml`:

```toml
[tool.furrow]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
max_parallel_tasks = 5
```

## Commands

- `furrow start <goal>` - Start the development loop
- `furrow web` - Start web UI (default: http://localhost:8000)
- `furrow init` - Initialize project with .env template

## Features

- **Infinite development loop** - Plans, executes, tests, and repeats until done
- **Parallel execution** - Runs up to 5 tasks concurrently
- **State persistence** - Automatically saves progress to `.furrow_state.json`
- **Resume capability** - Continue from where you left off with `--resume`
- **File operations** - Workers can read/write/list files directly
- **Structured logging** - Uses structlog for detailed observability
- **Retry logic** - Automatic retries with exponential backoff for API calls
- **Web UI** - Monitor progress in browser
- **Config file support** - furrow.toml or pyproject.toml

## Architecture

- **Orchestrator** - Runs the infinite loop, coordinates agents
- **Planner** - Breaks goals into parallelizable tasks
- **Worker** - Executes tasks with file system access
- **Tester** - Runs tests and reports failures

## Example

```bash
# In a Python project
furrow start "Add user authentication with JWT tokens"

# The agent will:
# 1. Plan tasks (e.g., create auth module, add tests, update requirements)
# 2. Execute tasks in parallel using file tools
# 3. Run pytest to verify
# 4. Repeat until all tests pass
```