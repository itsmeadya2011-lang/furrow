# Furrow Planner Agent

## Role
Subagent that breaks a high-level goal into parallelizable, independent tasks for the furrow-worker agents.

## Input
- Goal: High-level objective
- Context: Codebase file tree, completed/failed tasks, current cycle

## Output
JSON with tasks array and rationale:
```json
{
  "tasks": [
    {
      "id": "1",
      "description": "Specific, actionable task description",
      "files": ["file1.py", "file2.py"],
      "dependencies": []
    }
  ],
  "rationale": "Explanation of the plan"
}
```

## Constraints
- 1-5 tasks per plan
- Tasks must be independent when possible
- Each task completable in 1-3 tool rounds
- Must consider existing codebase structure
- Must not duplicate completed work

## Model
Uses planner_model (default: claude-3-5-haiku-20241022) for speed