---
description: Start the infinite Furrow development loop
agent: furrow
subtask: false
---
Start the Furrow development loop with the goal from $ARGUMENTS.
If no goal is provided, ask the user for one before starting.
This is the Kilo-session entry point: the loop keeps running until the goal is met or the user stops it, and the goal prompt is the only input it takes. The Python package in `furrow/` is separate and is driven by the `furrow start` / `furrow web` CLI.
