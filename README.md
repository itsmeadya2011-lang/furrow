# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

### CLI

```bash
furrow start "Add JWT auth to the API"
furrow start --model "gpt-4o" "Refactor the database layer"
```

### Web UI

```bash
furrow web
```

Then open http://localhost:8000 in your browser.

### Environment Variables

- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` - Required for LLM access
- `FURROW_PROVIDER` - `anthropic` (default) or `openai`
- `FURROW_MAX_CYCLES` - Maximum development cycles (0 = unlimited)
- `FURROW_MAX_PARALLEL_TASKS` - Max parallel workers (default: 5)

## How It Works

1. **Plan** - Breaks your goal into 1-5 parallelizable tasks using the project file tree for context
2. **Execute** - Workers implement each task by reading, modifying, and writing files
3. **Test** - Runs your test suite and analyzes results
4. **Repeat** - Fixes failures and continues until the goal is complete

## Architecture

- `PlannerAgent` - Analyzes the codebase and creates task plans
- `WorkerAgent` - Implements individual tasks with file operations
- `TesterAgent` - Runs tests and evaluates results
- `Orchestrator` - Manages the development loop
