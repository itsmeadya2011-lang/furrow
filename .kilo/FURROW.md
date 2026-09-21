# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs development cycles for web, CLI, and desktop projects. The configured `max_cycles` value governs the maximum number of cycles; `0` leaves the cycle cap unset.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent (also usable as a subagent)
- **Role:** Plans work, delegates tasks, verifies results, retains state, and retries failures within the cycle limit.
- **Cycle:**
  1. Receive or identify the current goal and retained task state.
  2. Plan 1-5 valid tasks, or return an explicit empty plan when no work is proposed.
  3. Execute planned tasks through `furrow-worker` agents.
  4. Verify the retained work with `furrow-tester`.
  5. Retry unfinished or failed work, preserving the original goal, until completion or `max_cycles`.

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, and `dependencies`.
- **Validation:** Use 1-5 tasks with unique IDs, nonempty descriptions, known dependency IDs, and an acyclic dependency graph. An explicit empty task list means no work is proposed, not success.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes one assigned task with minimal, targeted changes and no scope creep.
- **Constraint:** 1-3 tool call rounds per task.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs configured verification with a timeout and returns a summary plus useful failures.
- **Status:** Exit code 0 is pass; nonzero, timeout, missing runner, or execution error is fail. The tester diagnoses and summarizes but cannot override status.

## Runtime State and Stopping
- Task IDs, statuses, results, the original goal, latest verification, and cycle count are retained across cycles.
- Worker exceptions become failed tasks in retained state and are carried into retry planning.
- Verification failures retry from the original goal plus the latest failure details.
- An empty or no-progress plan is not completion. The run stops after a nonempty completed plan passes authoritative verification, or when `max_cycles` is reached.

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
- Run `/furrow <goal>` to start the cycle workflow.
- The agent stops on completion, at the configured `max_cycles`, or when the user stops the session.
- Mid-cycle user input is incorporated into the next plan.

## Future Work
- CLI and desktop wrappers for non-TUI interaction.
- Persistent state file for goal/task tracking across sessions.
- Web UI for monitoring parallel agent activity.
