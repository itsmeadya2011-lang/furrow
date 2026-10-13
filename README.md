# Furrow

Autonomous coding agent with an infinite parallel development loop.

## Install

```bash
pip install furrow
```

## Usage

```bash
# Start a development loop
furrow start "Add JWT auth to the API"

# Start with options
furrow start "Build a REST API" --max-cycles 10 --provider anthropic --model claude-3-5-sonnet-20241022

# Start web UI
furrow web
```

## Configuration

Set environment variables or create a `.env` file:

```bash
# Required: API key for your provider
export ANTHROPIC_API_KEY=your-key
# or
export OPENAI_API_KEY=your-key

# Optional settings
export FURROW_PROVIDER=anthropic
export FURROW_MODEL=claude-3-5-sonnet-20241022
export FURROW_MAX_CYCLES=0  # 0 = unlimited
export FURROW_LOG_LEVEL=INFO
export FURROW_LOG_FORMAT=json
```

## Commands

| Command | Description |
|---------|-------------|
| `furrow start <goal>` | Start autonomous development loop |
| `furrow web` | Start web UI on port 8000 |

## Options for `furrow start`

| Option | Description |
|--------|-------------|
| `--max-cycles N` | Maximum development cycles (0 = unlimited) |
| `--workspace PATH` | Working directory |
| `--provider PROVIDER` | LLM provider (anthropic, openai, ollama) |
| `--model MODEL` | Override LLM model |
| `--log-level LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `--log-format FORMAT` | Log format (json, console) |

## How It Works

Furrow runs an infinite loop until the goal is complete:

1. **Plan** - Break goal into parallelizable tasks
2. **Execute** - Run tasks in parallel using worker agents
3. **Test** - Run tests, linting, and type checking
4. **Repeat** - Fix failures and continue until done

Each worker agent reads relevant files, makes targeted changes, and writes them back.
