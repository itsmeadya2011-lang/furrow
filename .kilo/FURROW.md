# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the infinite loop. Receives goals, plans, delegates, tests, and repeats. Maintains state persistence across cycles.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` subagents
  4. Test via `furrow-tester` subagent
  5. Report and persist state
  6. Repeat

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks. Considers project file structure when planning.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Uses `read_file` and `write_file` tools to inspect and modify code. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task. Returns JSON with `files_changed`, `summary`, and `complete`.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs tests, lint, type checks. Fixes failures. Returns strict JSON with `passed`, `summary`, `failures`, and `files_checked`.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`. Passes `--max-cycles`, `--no-test`, `--workspace`, and `--verbose` flags.

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
kilo.json              # Project config (default agent = furrow, open permissions)
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## Future Work
- Persistent state file for goal/task tracking across sessions (now implemented via state persistence).
- Structured logging for all cycle events.
- Retry logic for failed workers (spawn replacement worker in the same cycle).
- Web UI improvements for monitoring parallel agent activity.

## Implementation Notes
- **State Persistence:** Goal and task state is written to the workspace after each cycle so sessions can resume.
- **Structured Logging:** Each cycle logs structured events for debugging and monitoring.
- **Retry Logic:** Failed workers are immediately replaced with fix instructions within the same cycle.
- **File Tool Usage:** Workers use `read_file` and `write_file` for code inspection and modification, avoiding shell commands unless explicitly required.
