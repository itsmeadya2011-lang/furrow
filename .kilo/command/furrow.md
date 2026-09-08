---
description: Start the infinite Furrow development loop
agent: furrow
subtask: false
---
Start the Furrow development loop with the goal from $ARGUMENTS.
If no goal is provided, ask the user for one before starting.

## Options
- `FURROW_MODEL` env var or `--model` flag overrides the LLM model.
- `FURROW_MAX_CYCLES` env var limits the number of cycles (0 = unlimited).
- `FURROW_WORKSPACE` env var sets the working directory.
