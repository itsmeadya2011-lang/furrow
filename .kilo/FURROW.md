# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the infinite loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Report and repeat

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.
- **Features:**
  - Inspects workspace structure before planning
  - Reads configuration files (pyproject.toml, package.json, etc.)
  - Provides context-aware task decomposition

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task by making actual file changes.
- **Constraint:** 1-3 tool call rounds per task.
- **Features:**
  - Reads relevant files before making changes
  - Creates, modifies, or deletes files as needed
  - Returns detailed summary of changes made

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Fixes failures. Returns pass/fail JSON.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

## Files
```
.kilo/
  agent/
    furrow.md          # Orchestrator
    furrow-planner.md  # Planner subagent
    furrow-worker.md   # Worker subagent
    furrow-tester.md   # Tester subagent
  command/
    furrow.md          # /furrow slash command
furrow/
  core/
    orchestrator.py    # Main loop controller
    state.py           # Persistent state management
  agents/
    planner.py         # Planning agent
    worker.py          # Worker agent
    tester.py          # Testing agent
    prompts.py         # LLM prompts
  config.py            # Settings and data models
  llm.py               # LLM client interface
kilo.json              # Project config (default agent = furrow, open permissions)
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## Configuration
Environment variables (prefix with `FURROW_`):
- `FURROW_PROVIDER`: LLM provider (anthropic, openai, ollama)
- `FURROW_MODEL`: Default model name
- `FURROW_MAX_PARALLEL_TASKS`: Maximum parallel tasks (default: 5)
- `FURROW_MAX_CYCLES`: Maximum cycles before stopping, 0 = infinite (default: 0)
- `FURROW_WORKSPACE`: Working directory (default: current directory)

## Error Handling
- Automatic retry with exponential backoff for planning failures
- Task-level retry for transient errors
- Graceful handling of keyboard interrupts
- Session state preserved on interruption

## State Management
- Session state saved to `.furrow_state.json` in workspace
- Tracks completed/failed tasks across sessions
- Allows resuming context from previous runs

## Future Work
- CLI and desktop wrappers for non-TUI interaction.
- Web UI for monitoring parallel agent activity.
- Integration with version control (git).
- Support for task dependencies and ordering.
