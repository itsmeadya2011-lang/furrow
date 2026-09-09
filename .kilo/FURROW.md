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

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.

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
kilo.json              # Project config (default agent = furrow, open permissions)
```

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete or the user stops the session.
- Mid-loop user input is incorporated into the next cycle's plan.

## Implementation Notes

### Orchestrator (`furrow/core/orchestrator.py`)
- Tracks `self.tasks` across cycles so `_is_done()` can inspect them. The previous
  `_get_tasks()` returned `[]` unconditionally, which made the loop halt after the
  first cycle regardless of outcome.
- `_is_done()` returns `False` while there are failed tasks (so fixes are attempted
  in the next cycle) and `True` only when the planner produces no tasks or all
  tracked tasks completed.
- Accepts an optional `on_output: Callable[[str]]` callback. When set, every status
  message is forwarded to it in addition to the console. This lets the web server
  stream live output to WebSocket clients.
- Preserves the original goal in `self.original_goal` for reporting.

### Worker (`furrow/agents/worker.py`)
- Workers no longer just echo LLM text. `run()` now:
  1. Reads each file in `self.task.files` via `LLMClient.read_file()` for context.
  2. Builds a prompt including the task, files to touch, and current file contents.
  3. Asks the LLM to return JSON: `{"files": [{"path", "content"}], "summary"}`.
  4. Writes each file via `LLMClient.write_file()`.
  5. Falls back to `### filename` markdown sections if JSON parsing fails.
- Returns the summary string (unchanged return type).

### LLM Client (`furrow/llm.py`)
- Added Ollama provider support. `Provider.OLLAMA` now routes to
  `_complete_ollama()`, which uses an OpenAI-compatible `AsyncOpenAI` client
  pointed at `ollama_base_url`. Anthropic and OpenAI paths are unchanged.

### Web Server (`furrow/web/server.py`)
- The `/ws` endpoint now runs the orchestrator in a background task and streams
  its `on_output` messages to the browser, stripping rich markup tags.
- The connection stays open until the orchestrator finishes or the client
  disconnects.

### CLI (`furrow/cli/`)
- Added the missing `__init__.py` so `furrow.cli` is a proper package.

## Future Work
- Persistent state file for goal/task tracking across sessions.
- Desktop wrappers for non-TUI interaction.
- Rate-limit awareness and retry handling in the LLM client.
