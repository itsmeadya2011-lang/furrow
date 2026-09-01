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
  furrow-state.json   # Persistent state (goal, completed/in-progress tasks, cycle)
  agent/
    furrow.md          # Orchestrator
    furrow-planner.md  # Planner subagent
    furrow-worker.md   # Worker subagent
    furrow-tester.md   # Tester subagent
  command/
    furrow.md          # /furrow slash command
kilo.json              # Project config (default agent = furrow, open permissions)
```

## State Management

The orchestrator persists loop state to `.kilo/furrow-state.json` so sessions can resume after restart.

**Lifecycle:**
1. **Load** — At the start of each cycle, the orchestrator reads the state file via the `read` tool. If the file does not exist, it initializes fresh state from the user's goal.
2. **Skip completed** — The planner excludes tasks already marked `completed` from new plans.
3. **Write** — After each cycle completes, the orchestrator writes the updated state via the `write` tool.

**Schema:**
```json
{
  "goal": "string — the current objective",
  "cycle": "integer — current loop iteration (1, 2, …)",
  "completedTasks": ["array of task IDs that finished successfully"],
  "inProgress": ["array of task IDs currently executing"],
  "updatedAt": "ISO 8601 timestamp of last write"
}
```

**Fields:**
- `goal` — Set once when the loop starts; persists across restarts.
- `cycle` — Increments after each plan→execute→test→report sequence.
- `completedTasks` — Used by the planner to avoid re-doing work.
- `inProgress` — Tracks tasks delegated to workers but not yet returned.
- `updatedAt` — Useful for debugging stale state.

## Permissions
Open by default: `bash: allow`, `edit: allow`, `read: allow`.

## Quickstart

Invoke the orchestrator from the TUI:

```
/furrow <goal>
```

Example:

```
/furrow Refactor the auth module to support OAuth2
```

The loop runs until the goal is complete, the user stops the session, or a non-recoverable error is encountered. Mid-loop user input is incorporated into the next cycle's plan.

## Plan Format

The planner outputs JSON consumed by the orchestrator. Each task is independent and assigned to a worker subagent.

```json
{
  "tasks": [
    {
      "id": "1",
      "description": "Extract JWT validation into a shared middleware",
      "files": ["src/auth/middleware.ts", "src/auth/__tests__/middleware.test.ts"],
      "dependencies": []
    },
    {
      "id": "2",
      "description": "Add refresh-token rotation and revocation endpoints",
      "files": ["src/auth/tokens.ts", "src/auth/__tests__/tokens.test.ts"],
      "dependencies": ["1"]
    }
  ]
}
```

Fields:
- `id` — unique task identifier
- `description` — human-readable objective
- `files` — list of files the worker may read or edit
- `dependencies` — array of `id`s that must complete first

## Worker Contract

Each `furrow-worker` subagent operates under a strict 1–3 tool call round limit per task:

1. The worker receives the task description and target files.
2. It may issue up to **3 tool call rounds** (e.g., `read`, `edit`, `bash`).
3. It must return a concise summary of changes, including any test commands run.
4. If the task is blocked by missing context or an external dependency, the worker should return the blocker rather than exceed the round limit.

This constraint prevents runaway agents and keeps work scoped to the planner's intent.

## Future Work

The following items are follow-up specs intended for upcoming planning cycles. Each is scoped but not yet committed for implementation.

### 1. CLI and Desktop Wrappers
**Scope**
- Provide non-TUI entry points to the Furrow orchestrator:
  - A standalone CLI binary that accepts a goal via flags or stdin and runs the same loop as the TUI.
  - A desktop wrapper (Tauri or Electron) that embeds a webview pointing at the future Web UI, with tray controls to start/stop the loop.
- Both wrappers reuse the existing `furrow` orchestrator logic; they are thin shells that adapt input/output and lifecycle rather than re-implementing the loop.
- Surface minimal controls: start/stop, current goal, latest plan summary, last test status.

**Prerequisites**
- Web UI spec (#3) at least stubbed so the desktop wrapper has a target to embed.
- A stable internal entry point in the orchestrator that does not depend on TUI-specific prompt state.
- Decision on packaging target (npm binary vs. native installer) and code-signing requirements.
- Permission/sandbox story for the desktop app on at least one target OS.

**Suggested First Slice**
- Extract a `runLoop(goal, options)` function from the TUI command handler so it can be invoked from non-TUI contexts.
- Build a minimal Node/TS CLI (`furrow-cli`) that calls `runLoop` with a goal passed via `--goal` and streams progress to stdout.
- Land this behind a feature flag; do not wire the desktop wrapper yet.

### 2. Persistent State (Partial)
**Status:** First slice implemented — orchestrator now loads/writes `.kilo/furrow-state.json` each cycle. See [State Management](#state-management) for the current schema and lifecycle.

**Remaining scope**
- Explicit resume semantics: on startup, detect an unfinished goal and offer to continue from the last completed task.
- `--resume` flag that loads state and prints the last plan.
- Concurrency model if multiple `furrow` processes could touch the same state file (single-writer lock or append-only journal).
- Schema migrations: add a `version` field and upgrade hooks.

**Original Scope**
- Persist orchestrator state across sessions so a goal and its plan survive crashes, restarts, and machine swaps.
- Track: active goal, planned tasks (id, description, files, dependencies, status), last known test result, current cycle index.
- Provide explicit resume semantics: on startup, detect an unfinished goal and offer to continue from the last completed task.
- Use a human-readable format (JSON or YAML) so the state file is easy to inspect and edit.

**Prerequisites**
- Stable task identity: tasks need durable IDs across planner invocations.
- Decision on storage location (project-local `.furrow/state.json` vs. user-global `~/.furrow/state.json`) and merge strategy if both exist.
- Concurrency model if multiple `furrow` processes could touch the same state file (single-writer lock or append-only journal).
- Migrations plan: state schema will evolve, so a `version` field and upgrade hooks are required from day one.

**Suggested Next Slice**
- Add a `--resume` flag that, when present, loads state and prints the last plan; resume logic itself lands in a later slice.
- Add a `version` field to the state schema and a migration helper.

### 3. Web UI
**Scope**
- A lightweight web dashboard for monitoring one or more Furrow loops in real time:
  - Active goal and current cycle.
  - Plan view: tasks with status (pending/in_progress/completed/failed), dependencies graph.
  - Live log stream from worker and tester subagents.
  - Manual controls: pause, resume, stop, inject a steering message.
- Read-only by default; write controls gated behind an explicit "operator mode" toggle to avoid accidental edits.
- Transport: server-sent events (SSE) or WebSocket; no polling.

**Prerequisites**
- Persistent state (#2) so the UI has a stable source of truth to render.
- A small HTTP server colocated with the orchestrator (or a sidecar) that exposes the state and event stream.
- Decision on frontend stack (vanilla TS + a single component lib, or a framework) and build tooling.
- Auth story, even if minimal (local-only token), before exposing anything beyond `localhost`.

**Suggested First Slice**
- Stand up a minimal Express/Fastify server that serves a static page reading from a stubbed in-memory state object.
- Render a single "current goal + task list" view; no live updates yet.
- Add SSE plumbing on a separate endpoint that pushes one synthetic event per second, just to validate the transport end-to-end before wiring real orchestrator events.
