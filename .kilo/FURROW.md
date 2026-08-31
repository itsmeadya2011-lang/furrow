# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs an infinite development loop for web/CLI/desktop projects. It plans parallel, independent tasks, executes them via worker subagents, validates with a tester, and repeats until the goal is verifiably complete.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as subagent)
- **Role:** Runs the infinite loop. Receives goals, invokes the planner, delegates to parallel workers, retries failures, runs the tester, persists state, and repeats.
- **Loop:**
  1. **Load State** - Read `.kilo/furrow-state.json`. If missing, initialize with `current_goal`, `cycle: 0`, and empty `pending`/`completed`/`failed` records.
  2. **Plan** - Invoke the `furrow-planner` subagent (`subtask: true`) with the current goal and state. It returns an ordered list of 1-5 tasks with `id`, `description`, `files`, `dependencies`, and `verification`, plus top-level `project_type`/`test_command`/`lint_command`. Replace `pending` with the new plan's task ids.
  3. **Execute** - For each task in `pending`, spawn a `furrow-worker` subagent (`subtask: true`) passing: task `id`, `description`, target `files`, `dependencies`, and `verification` command. Collect `changed_files`, `status`, and `issues` into state (`completed` or `failed`).
  4. **Retry** - For any task in `failed` with `retry_count < 2`, spawn a new `furrow-worker` with fix instructions derived from the prior failure output. Increment `retry_count`. After retries exhausted, mark the task as permanently failed for the cycle.
  5. **Test** - Spawn `furrow-tester` (`subtask: true`) with the goal, the list of changed files from completed tasks, and the plan's verification/test commands. If it reports failures, spawn targeted `furrow-worker` fix agents per failing area. Repeat until tests pass or fix retries are exhausted.
  6. **Report** - Output a structured cycle report (Cycle, Goal, Tasks Completed/Failed, Test Result, Remaining Work, Next Action).
  7. **Decide** - If all planned tasks completed, tests pass, and no outstanding todos remain, declare the goal complete and stop. Otherwise, increment `cycle`, persist state, and start the next cycle.

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into 1-5 parallelizable, independent tasks. Inspects the repository to detect the project type and toolchain before planning.
- **Output:** JSON object (not wrapped in markdown) with:
  - `project_type`, `test_command`, `lint_command` (top-level, used for auto-detection fallbacks)
  - `tasks[]` (each with `id`, `description`, `files`, `dependencies`, `verification`)
  - `rationale`, `multi_cycle`, `next_cycle_goal`

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Behavior:**
  - Reads each target file before editing (read-before-edit discipline) and confirms it exists.
  - Runs the planner-provided `verification` command after changes; falls back to a sensible project-type default if none provided.
  - On verification failure, attempts a direct minimal fix within the same round (no sub-spawns). Returns status `failed` with details if still failing.
- **Return:** Structured JSON with `task_id`, `status` (`complete`|`failed`), `changed_files`, `summary`, `issues[]`.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Auto-detects the project toolchain, runs the test/lint/typecheck suite, and distinguishes preexisting failures from cycle-introduced ones. Fixes cycle failures; reports preexisting ones without fixing.
- **Behavior:**
  - Detection order: planner-provided `test_command` > `package.json` > `pyproject.toml`+`Makefile` > `Makefile` > `go.mod` > `Cargo.toml` > `pom.xml` > `build.gradle` > `*.csproj`. Prioritizes `lint` -> `type-check` -> `test`.
  - Tracks every command executed in a `commands_run` array.
  - On failure, identifies the offending file and whether it changed this cycle. Fixes only cycle-introduced failures with minimal changes; reports preexisting failures in `failures` with `status: "preexisting"`.
- **Return:** JSON with `passed`, `summary`, `commands_run`, `failures[]` (each with `file`, `line?`, `message`, `command`, `status?`).

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.
- **Usage:**
  - `/furrow <goal>` - Start the development loop with the given goal.
  - `/furrow --help` - Show a brief help message describing the command and its arguments.
  - `/furrow` (no args) - Prompt for a goal inline before starting.

## Files
```
.kilo/
  agent/
    furrow.md          # Orchestrator (primary)
    furrow-planner.md  # Planner subagent
    furrow-worker.md   # Worker subagent
    furrow-tester.md   # Tester subagent
  command/
    furrow.md          # /furrow slash command
  furrow-state.json    # Runtime state: goal, cycle, pending/completed/failed tasks (resumable across sessions)
kilo.json              # Project config (default agent = furrow, open permissions)
```
Global config: `~/.config/kilo/` (same structure; loads default agent, commands, and permissions).

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete or the user stops the session.
- A persistent state file at `.kilo/furrow-state.json` tracks progress across cycles, making the loop resumable without an explicit resume command.
- Run `/furrow --help` for a brief help message on the command's arguments.
- Mid-loop user input is incorporated into the next cycle's plan.

Examples:
- `/furrow "Add JWT auth to the API"`
- `/furrow "Migrate the CLI to use commander.js"`
- `/furrow "Add unit tests for the payment service and make them green"`
- `/furrow "Refactor the data layer to support multiple database backends"`

## Future Work
- CLI and desktop wrappers for non-TUI interaction.
- Web UI for monitoring parallel agent activity.
- Multi-project worktrees with Agent Manager integration (isolated worktrees per goal, managed via `agent-manager.json`).
- Per-cycle git commit snapshots (snapshot changed files each cycle for easy rollback and history).
