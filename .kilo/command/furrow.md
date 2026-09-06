---
description: Start the infinite Furrow development loop
agent: furrow
subtask: false
---
Start the Furrow development loop with the goal from $ARGUMENTS.
If no goal is provided, ask the user for one before starting.

## Initialization
- Before starting the loop, check if `.furrow-state.json` exists in the project root.
- If it does not exist, create it with:
  ```json
  {
    "current_goal": "<goal from $ARGUMENTS>",
    "completed_tasks": [],
    "pending_tasks": [],
    "last_cycle_result": null,
    "iteration_count": 0,
    "user_input_log": []
  }
  ```
- If it does exist, load it and use the persisted `current_goal` as the starting goal (unless a new goal was passed via `$ARGUMENTS`, in which case update `current_goal`).
