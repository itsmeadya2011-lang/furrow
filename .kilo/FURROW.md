# Furrow Agent Architecture

## Overview

Furrow is a bounded, verification-focused Kilo workflow for web, CLI, and desktop projects. The Kilo orchestrator runs finite development waves; it is not an infinite loop. The Python Furrow package is a separate runtime.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent
- **Role:** Preserves the original goal, owns task state and the finite cycle budget, delegates waves through the Kilo Task tool, verifies results, and decides when to stop.
- **Wave:** Plan 1-5 tasks, execute the batch, run focused verification, update state, then either continue or stop.

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a goal into small, parallelizable tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, and `dependencies`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes one assigned task with minimal, targeted changes and no scope creep.
- **Constraint:** 1-3 tool-call rounds per task; does not spawn subagents.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs focused tests, lint, and type checks, diagnoses failures, and returns pass/fail evidence. Any fixes are narrowly scoped and must be verified by the parent.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point that invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

## Workflow

1. Anchor the user's goal verbatim as `original_goal` and set a finite cycle budget (default: 5).
2. Plan 1-5 dependent-aware tasks, prioritizing the smallest verifiable slice.
3. Delegate the wave with Kilo's `task` tool and `subagent_type` set to the configured Furrow subagent. Keep dependent work foreground; use `background: true` only for independent work and wait before integrating it.
4. After each worker wave, run focused checks for changed and affected files, then run the relevant broader suite when appropriate.
5. Update task states and preserve `original_goal`; turn failures into targeted remediation waves rather than replacing the goal.
6. Stop on verified completion, user interruption, budget exhaustion, or 2 consecutive no-progress cycles. Report incomplete work and blockers honestly.

## State And Safety

Maintain a compact state with the original/current goal, cycle number and limit, task ids/statuses/dependencies/results, verification evidence, blockers, retry count, and no-progress count. Use `pending`, `in_progress`, `completed`, `failed`, and `blocked` consistently.

A no-progress cycle makes no verified task progress, produces no meaningful diff, repeats the same failure, or repeats a plan without changing state. Stop after two such consecutive cycles. Never report completion from a child summary alone: verify acceptance criteria, checks, and the final diff first.

## Runtime Boundary

The **Kilo Task-tool loop** and the **Python Furrow runtime** are distinct:

- A Kilo Task-tool child is an isolated Kilo session that shares the parent's workspace; it is not a Python subprocess.
- The Python package's `furrow.core.orchestrator.Orchestrator`, `PlannerAgent`, `WorkerAgent`, and `TesterAgent` form a separate implementation with its own loop and settings.
- Do not recursively invoke `/furrow` from a child, and do not treat Task-tool delegation as launching the Python runtime. These documentation instructions govern the Kilo workflow only.

## Files

```text
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

- Run `/furrow <goal>` to start the bounded Kilo workflow.
- The workflow stops when its goal is verified complete, its budget or no-progress limit is reached, or the user stops the session.
- Mid-loop user input is recorded as an explicit goal update while the original goal remains preserved.
- `furrow start` in the Python package is a separate runtime entry point, not the Kilo Task-tool loop.

## Future Work

- CLI and desktop wrappers for non-TUI interaction.
- Persistent state file for goal/task tracking across sessions.
- Web UI for monitoring parallel agent activity.
