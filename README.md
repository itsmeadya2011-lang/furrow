# Furrow

**Furrow** is an autonomous coding agent that runs an infinite parallel development loop. It breaks down high-level goals into parallel tasks, executes them via specialized subagents, runs tests and linting, and repeats until the goal is complete.

## Features

- **Infinite development loop** — Plans, executes, tests, and iterates automatically
- **Parallel task execution** — Up to 5 workers run concurrently
- **Multi-provider LLM support** — Anthropic, OpenAI, and Ollama
- **Real-time web UI** — Monitor agent activity in the browser
- **CLI and slash command** — Run from terminal or Kilo TUI
- **Structured output** — Workers produce JSON, enabling programmatic use

## Installation

### Development Install (recommended)

```bash
git clone <repo-url>
cd furrow
pip install -e ".[dev]"
```

This installs Furrow in editable mode with development dependencies (pytest, ruff, mypy).

### Production Install

```bash
pip install furrow
```

## Quick Start

```bash
# Set your API key
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"

# Run a goal via CLI
furrow start "Add JWT authentication to the FastAPI backend"

# Run with custom settings
furrow start "Refactor the user service" --cycles 10 --provider anthropic --model claude-3-5-sonnet-20241022

# Start the web UI
furrow web
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | For Anthropic provider | Your Anthropic API key |
| `OPENAI_API_KEY` | For OpenAI provider | Your OpenAI API key |
| `OLLAMA_HOST` | For Ollama provider | Ollama server URL (default: http://localhost:11434) |

## Supported Providers

- **Anthropic** — Claude 3.5 Sonnet, Opus, Haiku
- **OpenAI** — GPT-4o, GPT-4o-mini, o1 series
- **Ollama** — Local models (llama3, codellama, etc.)

## CLI Reference

```
furrow start <goal>     Run the autonomous loop with a goal
furrow web              Start the web UI server
furrow --help           Show all options
```

### `furrow start` Options

| Flag | Description | Default |
|------|-------------|---------|
| `--cycles`, `-c` | Maximum development cycles | 10 |
| `--provider`, `-p` | LLM provider (anthropic, openai, ollama) | anthropic |
| `--model`, `-m` | Specific model name | Provider default |
| `--parallel`, `-j` | Max parallel workers | 3 |

## The Infinite Development Loop

```
┌─────────────┐
│  Goal Input │
└──────┬──────┘
       ▼
┌─────────────┐     ┌──────────────┐
│  Planner    │────▶│  Parallel    │
│  (subagent) │     │  Workers     │
└─────────────┘     └──────┬───────┘
                           ▼
                    ┌─────────────┐
                    │   Tester    │
                    │ (subagent)  │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  Complete?  │──No──▶ (repeat)
                    └──────┬──────┘
                           │Yes
                           ▼
                    ┌─────────────┐
                    │   Done!     │
                    └─────────────┘
```

The loop continues until:
- The goal is marked complete by the orchestrator
- `max_cycles` is reached
- The user interrupts (Ctrl+C)

## Development

### Prerequisites

- Python 3.10+
- uv (recommended) or pip

### Setup

```bash
# Clone and enter
git clone <repo-url>
cd furrow

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check .

# Run type checker
mypy furrow
```

### Project Structure

```
furrow/
├── cli/           # CLI entry point
├── core/          # Orchestrator and loop logic
├── agents/        # Planner, Worker, Tester subagents
├── web/           # FastAPI web UI
├── llm.py         # LLM provider abstraction
└── config.py      # Configuration management
```

## License

MIT