# Furrow Agent Architecture

## Overview
Furrow is an autonomous coding agent that runs a bounded development loop for web/CLI/desktop projects. Cycles are bounded by `max_cycles` (default: 25; 0 = unlimited). Each cycle includes explicit test gates and verification before proceeding.

## Components

### 1. Orchestrator (`furrow`)
- **Type:** Primary agent
- **Role:** Runs the development loop. Preserves the original goal, plans tasks respecting dependencies, delegates to workers, tracks task state, emits web progress, tests, and repeats up to `max_cycles`.
- **Loop:**
  1. Receive / Identify goal
  2. Plan parallel tasks (1-5)
  3. Execute via parallel `furrow-worker` agents
  4. Test via `furrow-tester` agent (explicit gate)
  5. Verify results; if tests fail, failure context is passed to planning while the original goal remains unchanged, then fix and retry (bounded by `max_cycles`)
  6. Report completion; halt when goal is complete or cycle limit reached

### 2. Planner (`furrow-planner`)
- **Type:** Subagent
- **Role:** Breaks a high-level goal into parallelizable, independent tasks.
- **Output:** JSON with `tasks[]`, each having `id`, `description`, `files`, `dependencies`.

### 3. Worker (`furrow-worker`)
- **Type:** Subagent
- **Role:** Executes a single assigned task. Minimal, targeted changes. No scope creep.
- **Constraint:** 1-3 tool call rounds per task.
- **Protocol:** Structured safe file-edit protocol — read before editing, make minimal changes, never delete unrelated code. Do not run shell commands directly.

### 4. Tester (`furrow-tester`)
- **Type:** Subagent
- **Role:** Runs the configured test command and reports failures. Does not run arbitrary lint/type checks unless configured. Does NOT edit source files — failures are handled by the orchestrator in the next cycle.
- **Output:** JSON pass/fail with summary and failure list.

### 5. Command (`/furrow`)
- **Type:** Slash command
- **Role:** Entry point. Invokes the `furrow` orchestrator with a goal from `$ARGUMENTS`.

## Workflow Safety

### Bounded Cycles
- The loop runs at most `max_cycles` times (default: 25; 0 = unlimited).
- Each cycle must produce a test result before proceeding.
- If `max_cycles` is reached without completion, the orchestrator halts and reports.

### Explicit Completion / Test Gates
- After each cycle, the tester runs the configured test command and reports results. A failed test does not block the next retry cycle; it supplies failure context while the original goal remains unchanged.
- If tests fail, failure context is passed to planning while the original goal remains unchanged; the next cycle attempts fixes. The orchestrator tracks dependencies and task state across cycles.
- Completion is verified: all tasks completed, no failures, tests pass.

### Safe File-Edit Protocol
- Workers use structured file operations: read before editing, make minimal changes, never delete unrelated code. Workers do not run tests; the orchestrator's tester runs verification after worker execution.

### Verification
- Every completed task is verified by the tester in the same cycle.
- End-to-end verification runs after all tasks complete.
- If verification fails, failure context is passed to planning while the original goal remains unchanged; the cycle repeats to attempt fixes.

### Cancellation
- The loop stops when: (a) goal is complete and tests pass, (b) `max_cycles` is reached, (c) no further progress is made, or (d) the user explicitly stops.
- No cycle should start if the goal is already complete or no progress has been made in recent cycles.
- On user cancellation (Ctrl+C), stop immediately and report progress.

### Least-Privilege Permissions
- Workers use structured file operations and may not directly run shell commands.
- The orchestrator tracks task dependencies and state across cycles.
- No worker spawns subagents.
- No worker modifies files outside its assigned scope.
- Permissions should be restricted to only what is needed for the current task.

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
  kilo.json              # Project config (permissions, default agent)
```

## Permissions
Default: `bash: ask`, `edit: ask`, `read: allow`. Consider restricting to least-privilege per task.

### Web UI
- Defaults to localhost.
- Streams events.
- Optional auth via `FURROW_WEB_TOKEN` or `web_token`.

## Usage
- Run `/furrow <goal>` to start the loop.
- The agent runs until the goal is complete, the cycle limit is reached, or the user stops.
- Mid-loop user input is incorporated into the next cycle's plan.

## Limitations
- Tester reports failures and does not edit source files.
- No persistent state across sessions; state is held in-memory during the loop.
