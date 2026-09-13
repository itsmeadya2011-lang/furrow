---
description: Start the bounded Furrow development loop
agent: furrow
subtask: false
---
Start the Furrow development loop with the goal from $ARGUMENTS.
If no goal is provided, ask the user for one before starting.

The loop runs bounded by `max_cycles` (default: 25; 0 = unlimited). Each cycle includes a test gate. The orchestrator halts when the goal is complete, no progress is made, or the cycle limit is reached. Web UI defaults to localhost; set `FURROW_WEB_TOKEN` (or `web_token`) to enable auth.
