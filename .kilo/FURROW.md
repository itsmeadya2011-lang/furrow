# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the infinite loop. Receives goals, plans, delegates, tests, and repeats.
- **Loop:**
   1. Receive / Identify goal
   2. Plan parallel tasks (1-5) with previous cycle context
   3. Execute via parallel `furrow-worker` subagents
   4. Test via `furrow-tester` subagent
   5. Report and repeat until planner returns no tasks

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks. Receives previous task results and cycle history.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task with file context. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Fixes failures. Returns pass/fail JSON with detailed failures.

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
  config.py            # Settings and models
  core/
    orchestrator.py    # Main loop with state tracking
  agents/
    planner.py         # Planning with context
    worker.py          # Execution with file reads
    tester.py          # Testing with diagnostics
    prompts.py         # LLM prompts
  llm.py               # LLM client with retry logic
  cli/
    main.py            # CLI entry point
  web/
    server.py          # Web UI
kilo.json              # Project config
```

## State Management
- Cycles are tracked in-memory and persisted to `.furrow_state.json`.
- Previous task results and test history are passed to the planner each cycle.
- The loop stops when the planner returns no tasks (goal complete).

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.
- Use `--state-file` to resume from a saved state.

## Future Work
- CLI and desktop wrappers for non-TUI interaction.
- Web UI for monitoring parallel agent activity.
