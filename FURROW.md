# Furrow Architecture

## Overview

Furrow is an autonomous development loop that iteratively plans, executes, and tests code changes to achieve a high-level goal. The orchestrator runs in cycles, each producing a verifiable increment toward the goal.

## Orchestrator Workflow

The orchestrator is a **primary agent** (mode: `primary`). It is not usable as a subagent. It follows a fixed loop that repeats until the goal is complete or max retries are exhausted:

```
Receive → Plan (via furrow-planner) → Execute (parallel furrow-worker) → Test (via furrow-tester) → Retry/Recover → Persist State → Report → Repeat
```

Each cycle:
1. **Plan** — Delegates to the `furrow-planner` subagent to decompose the goal into a prioritized task list.
2. **Execute** — Spawns `furrow-worker` subagents in parallel, one per task.
3. **Test** — Invokes the `furrow-tester` subagent to validate changes against acceptance criteria.
4. **Retry/Recover** — If a task or test fails, retries (up to the per-task limit) or re-plans.
5. **Persist State** — Atomically writes progress to `.kilo/furrow-state.json`.
6. **Report** — Summarizes the cycle and repeats.

**State:** Writes `.kilo/furrow-state.json` every cycle.

**Retry:** Max 3 retries per task, shared across worker and test failures.

**Multi-cycle:** When the planner returns `multi_cycle: true`, the orchestrator continues to the next cycle.

## Components

### 1. Orchestrator — `furrow.md`
**Type:** Primary agent only (mode: `primary`). NOT usable as a subagent.

**Role:** Coordinates the full loop, handles retries, decides continuation, and persists state.

**Loop:** Receive → Plan (via `furrow-planner`) → Execute (parallel `furrow-worker`) → Test (via `furrow-tester`) → Retry/Recover → Persist State → Report → Repeat.

**Behavior:**
- Reads and writes `.kilo/furrow-state.json` every cycle.
- **Retry policy:** Max 3 retries per task. The retry counter (`retries_used`) is shared across worker failures and test failures for the same task.
- **Multi-cycle:** When the planner returns `multi_cycle: true`, continues to a new cycle (re-planning any remaining tasks).
- **Recovery on restart:** Loads `.kilo/furrow-state.json`, resumes from remaining tasks.

---

### 2. Planner — `furrow-planner.md`
**Type:** Subagent (mode: `subagent`).

**Role:** Analyzes the goal and current state, produces an ordered task list.

**Input:** `goal` (string) and current state from `.kilo/furrow-state.json`.

**Output** (`plan.json`):
```json
{
  "tasks": [
    {
      "id": "string",
      "description": "string",
      "files": ["string"],
      "dependencies": ["string"],
      "priority": "high | medium | low",
      "effort": 1,
      "acceptance_criteria": "string"
    }
  ],
  "rationale": "string",
  "multi_cycle": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `tasks[]` | array | Ordered list of decomposed tasks |
| `tasks[].id` | string | Unique task identifier |
| `tasks[].description` | string | What the task does |
| `tasks[].files` | array[string] | Files the task touches |
| `tasks[].dependencies` | array[string] | IDs of tasks that must finish first |
| `tasks[].priority` | enum | `high`, `medium`, `low` |
| `tasks[].effort` | number | Estimated effort (1 = small, 3 = large) |
| `tasks[].acceptance_criteria` | string | Verifiable completion condition |
| `rationale` | string | Planner's reasoning for the plan |
| `multi_cycle` | boolean | `true` if the goal spans multiple cycles |

---

### 3. Worker — `furrow-worker.md`
**Type:** Subagent (mode: `subagent`).

**Role:** Executes one assigned task, producing file changes. One worker instance is spawned per task, running in parallel with other workers.

**Constraint:** 1–3 tool call rounds.

**Output** (`worker-result.json`):
```json
{
  "task_id": "string",
  "completed": true,
  "changes": [
    { "path": "string", "content": "string" }
  ],
  "files_modified": ["string"],
  "issues": ["string"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | ID of the task this result is for |
| `completed` | boolean | Whether the task succeeded |
| `changes[]` | array[object] | File changes made (path + content) |
| `files_modified[]` | array[string] | Paths of files actually written |
| `issues[]` | array[string] | Problems or warnings encountered |

---

### 4. Tester — `furrow-tester.md`
**Type:** Subagent (mode: `subagent`).

**Role:** Runs the test suite, validates acceptance criteria, and produces a pass/fail verdict.

**Output** (`test-result.json`):
```json
{
  "passed": true,
  "summary": "string",
  "failures": [
    {
      "task_id": "string",
      "description": "string",
      "fix_suggestion": "string"
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `passed` | boolean | Overall pass/fail verdict |
| `summary` | string | Human-readable summary of results |
| `failures[]` | array[object] | Details on failures. Each object has: `task_id`, `description`, `fix_suggestion` |

---

### 5. Command — `/furrow.md`
**Type:** Slash command (frontmatter: `agent: furrow`, `subtask: false`).

**Role:** Entry point invoked by the user. Resolves the active goal and delegates to the `furrow` orchestrator agent.

**Goal resolution order:**
1. If `$ARGUMENTS` is provided → use it as the active goal.
2. Else if `.kilo/furrow-state.json` exists → load and **resume** the saved goal.
3. Else → prompt the user for a goal.

After resolution, hands off to the `furrow` orchestrator agent to begin the cycle.

---

### 6. State File — `.kilo/furrow-state.json`
**Type:** Persistent JSON state file, read/written by the orchestrator every cycle. Enables resume across sessions.

## State File Format

Path: `.kilo/furrow-state.json`

```json
{
  "goal": "string",
  "current_cycle": 1,
  "multi_cycle": false,
  "tasks": [
    {
      "id": "string",
      "description": "string",
      "priority": "high | medium | low",
      "effort": 1,
      "acceptance_criteria": "string",
      "status": "pending | in_progress | completed | failed",
      "retries_used": 0,
      "files": ["string"],
      "dependencies": [],
      "worker_result": {
        "task_id": "string",
        "completed": true,
        "changes": [
          { "path": "string", "content": "string" }
        ],
        "files_modified": ["string"],
        "issues": ["string"]
      }
    }
  ],
  "cycle_history": [
    {
      "cycle": 1,
      "plan_rationale": "string",
      "multi_cycle": false,
      "tasks_completed": ["string"],
      "tasks_failed": ["string"],
      "worker_results": [],
      "test_result": {
        "passed": true,
        "summary": "string",
        "failures": []
      },
      "passed": true
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `goal` | string | High-level objective |
| `current_cycle` | number | The cycle currently in progress |
| `multi_cycle` | boolean | Whether the goal spans multiple cycles |
| `tasks[]` | array | All tasks with full status and latest worker result |
| `tasks[].status` | enum | `pending`, `in_progress`, `completed`, `failed` |
| `tasks[].retries_used` | number | Retries consumed for this task |
| `tasks[].worker_result` | object | Last worker output for the task |
| `cycle_history[]` | array | Audit trail of completed cycles |
| `cycle_history[].tasks_completed` | array[string] | Task IDs completed this cycle |
| `cycle_history[].tasks_failed` | array[string] | Task IDs that failed this cycle |
| `cycle_history[].test_result` | object | Tester output for the cycle |
| `cycle_history[].passed` | boolean | Whether the cycle's test passed |

## JSON Contracts Between Components

| From → To | Output shape | Key fields |
|-----------|-------------|------------|
| Planner → Orchestrator | `plan.json` | `tasks[]` (id, description, files, dependencies, priority, effort, acceptance_criteria), `rationale`, `multi_cycle` |
| Worker → Orchestrator | `worker-result.json` | `task_id`, `completed`, `changes[]`, `files_modified[]`, `issues[]` |
| Tester → Orchestrator | `test-result.json` | `passed`, `summary`, `failures[]` (objects: task_id, description, fix_suggestion) |
| Orchestrator → State File | `.kilo/furrow-state.json` | `goal`, `current_cycle`, `multi_cycle`, `tasks[]`, `cycle_history[]` |

## Retry Policy

- **Limit:** Max 3 retries per task.
- **Shared counter:** The retry counter (`tasks[].retries_used`) is shared across worker failures and test failures for the same task.
- **On failure:** When a task fails (worker `completed: false`) or a test fails (`tester.passed: false`), the task's `retries_used` is incremented.
- **Retry action:** The orchestrator re-runs the worker (and, if needed, the tester) up to the per-task limit of 3.
- **After 3 retries:** The task is marked `failed` and recorded in `cycle_history[].tasks_failed`. At the next cycle boundary, the planner is re-invoked with the failure context to re-plan.

## Multi-Cycle Handling

When the planner sets `multi_cycle: true`, the goal spans multiple cycles:

1. The planner produces an initial task breakdown, which may be a partial plan for the full goal.
2. Each cycle completes a subset of tasks; completed task IDs are recorded in `cycle_history[].tasks_completed`.
3. At the cycle boundary, the orchestrator persists state to `.kilo/furrow-state.json` and begins a new cycle.
4. The planner is re-invoked to:
   - Re-prioritize remaining tasks
   - Add discovered sub-tasks
   - Adjust for test feedback
5. The goal completes when all tasks are `completed`, `multi_cycle` is no longer required, and the tester passes.

## Workflow

The full execution flow:

```
                      ┌─────────────┐
                      │  /furrow    │
                      │  command    │
                      └──────┬──────┘
                             │ resolve goal
                             ▼
                  ┌───────────────────────┐
                  │   furrow.md           │
                  │   Orchestrator        │
                  │   (mode: primary)     │
                  └──────────┬────────────┘
                             │ Plan
                             ▼
                  ┌───────────────────────┐
                  │ furrow-planner.md     │
                  │  (mode: subagent)    │
                  │  → tasks[] (id,      │
                  │    description,      │
                  │    files, deps,      │
                  │    priority, effort, │
                  │    acceptance_        │
                  │     criteria),       │
                  │    rationale,        │
                  │    multi_cycle       │
                  └──────────┬────────────┘
                             │ Execute (parallel)
                             ▼
                  ┌───────────────────────┐
                  │ furrow-worker.md      │
                  │  (mode: subagent)    │
                  │  → task_id,          │
                  │    completed,        │
                  │    changes[],        │
                  │    files_modified[], │
                  │    issues[]          │
                  │  Constraint:         │
                  │   1–3 rounds         │
                  └──────────┬────────────┘
                             │ Test
                             ▼
                  ┌───────────────────────┐
                  │ furrow-tester.md      │
                  │  (mode: subagent)    │
                  │  → passed, summary,  │
                  │    failures[]        │
                  │    (task_id, desc,   │
                  │     fix_suggestion)  │
                  └──────────┬────────────┘
                             │
              ┌─────────────┴──────────────┐
              │ pass?                      │
              └──────┬─────────────┬───────┘
           yes      │             │     no / retries < 3
                     ▼             ▼
              ┌──────────┐  ┌──────────────┐
              │ Done /   │  │ Retry /      │
              │ Complete │  │ Recover      │
              │ Goal     │  └──────────────┘
              └──────────┘        │
                                  ▼
                          ┌────────────────┐
                          │  Persist State │
                          │ .kilo/furrow-  │
                          │ state.json     │
                          └───────┬────────┘
                                  │ multi_cycle?
                         ┌────────┴─────────┐
                         │      Report      │
                         └────────┬─────────┘
                                  │
                         ┌────────▼────────┐
                         │    Repeat       │
                         │  (next cycle)   │
                         │  if remaining   │
                         │  tasks          │
                         └────────┬────────┘
                                  ▼
                          ┌────────────────┐
                          │ cycle_history[]│
                          │ (audit trail)  │
                          └────────────────┘
```

## Files

```
.kilo/
├── furrow.md                  # Orchestrator agent (mode: primary) — full loop coordinator
├── furrow-planner.md          # Planner subagent — produces plan.json
├── furrow-worker.md           # Worker subagent — executes one task (1-3 rounds)
├── furrow-tester.md           # Tester subagent — validates changes
├── furrow-state.json          # Persistent state file (read/written every cycle)
└── FURROW.md                  # This documentation
/furrow.md                     # Slash command entry point (agent: furrow, subtask: false)
```

## Permissions

Open by default. The orchestrator and workers have full read/write access to the workspace. No sandboxing or permission prompts unless explicitly configured.

## Usage

```bash
# Start a new goal (goal = $ARGUMENTS)
/furrow "Build a REST API with authentication"

# Resume from saved state (goal loaded from .kilo/furrow-state.json)
/furrow

# The slash command resolves the goal:
#   1. $ARGUMENTS as the active goal
#   2. Load .kilo/furrow-state.json if it exists (resume)
#   3. Prompt the user if neither is available
#   4. Delegate to the furrow orchestrator agent
```

## Future Work

- CLI wrapper for non-Kilo environments
- Desktop app with live cycle visualization
- Web UI for remote monitoring and intervention
