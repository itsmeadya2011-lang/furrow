# Furrow

Autonomous coding agent with an infinite parallel development loop. Furrow breaks down your goals into tasks, executes them in parallel, tests the results, and iterates until the goal is achieved.

## How It Works

Furrow uses a continuous development cycle:

1. **Plan**: Analyze the goal and break it into 1-5 parallelizable tasks
2. **Execute**: Workers implement each task simultaneously using file operations
3. **Test**: Run tests to verify the implementation
4. **Iterate**: If tests fail, create a fix plan and repeat

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Planner   │────▶│  Workers    │────▶│   Tester    │
│  (1 cycle)  │     │ (parallel)  │     │  (verify)   │
└─────────────┘     └─────────────┘     └─────────────┘
       ▲                                      │
       └─────────── Fix Loop ◀────────────────┘
```

## Install

```bash
pip install furrow
```

## Quick Start

### Command Line

```bash
# Set your API key
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"

# Start Furrow with a goal
furrow start "Add JWT authentication to the API"

# Or use interactive mode
furrow start
```

### Web Interface

```bash
# Launch the web UI
furrow web
# Open http://localhost:8000 in your browser
```

## Configuration

All settings can be configured via environment variables with the `FURROW_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `FURROW_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai` |
| `FURROW_MODEL` | `claude-sonnet-4-20250514` | Default model for all agents |
| `FURROW_PLANNER_MODEL` | `claude-3-5-haiku-20241022` | Model for planning (faster/cheaper) |
| `FURROW_WORKER_MODEL` | `claude-3-5-sonnet-20241022` | Model for implementation |
| `FURROW_TESTER_MODEL` | `claude-3-5-sonnet-20241022` | Model for testing |
| `FURROW_MAX_PARALLEL_TASKS` | `5` | Maximum concurrent workers |
| `FURROW_MAX_CYCLES` | `0` | Max development cycles (0=unlimited) |
| `FURROW_WORKSPACE` | Current directory | Working directory for file operations |
| `FURROW_LOG_LEVEL` | `INFO` | Logging level |

### Using a `.env` File

Create a `.env` file in your project root:

```env
FURROW_PROVIDER=anthropic
FURROW_WORKER_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-...
```

## Architecture

### Agents

- **PlannerAgent**: Analyzes goals and creates task breakdowns
- **WorkerAgent**: Implements tasks by writing/editing files
- **TesterAgent**: Runs tests and evaluates results

### File Operations

Workers can perform three types of file operations:

```json
{
  "operations": [
    {"action": "write", "path": "file.py", "content": "..."},
    {"action": "edit", "path": "file.py", "old_text": "...", "new_text": "..."},
    {"action": "create_directory", "path": "src/utils"}
  ]
}
```

### Git Integration

Furrow automatically tracks all changes using git:

- Initializes a git repository if one doesn't exist
- Creates an initial commit before starting
- Commits changes after each cycle with descriptive messages
- Tags checkpoints for easy rollback

Disable git integration with `enable_git=False` when creating an Orchestrator.

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=furrow --cov-report=term-missing
```

## Limitations

- Workers operate on text-based file operations only (no shell execution)
- Git integration tracks changes but does not handle merge conflicts automatically
- Test runner auto-detection supports: pytest, npm/pnpm/yarn test, cargo test, go test
- Large codebases may require more specific file path guidance in goals

## License

MIT
