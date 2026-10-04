---
description: Start the bounded Furrow development loop
agent: furrow
subtask: false
---
Start the Furrow development loop with the goal from $ARGUMENTS.
If no goal is provided, ask the user for one before starting.

The loop runs until the goal is met or one of its bounds trips — it is not
endless. Follow the stopping rules in the `furrow` agent definition; do not
restart the loop after a stopping rule fires. Before starting, state which
provider and models will be used, so a misconfiguration is visible before the
first API call.