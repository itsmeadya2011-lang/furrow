---
description: Start the infinite Furrow development loop
agent: furrow
subtask: false
---
If a goal is provided via $ARGUMENTS, set it as the active goal.

If no goal is provided, load the current goal from `.kilo/furrow-state.json` (resuming previous session).

If neither a goal argument nor a saved state exists, ask the user for a goal before starting.

Hand off to the `furrow` orchestrator, which begins its planning → execution → testing cycle.