---
description: Start the infinite Furrow development loop
agent: furrow
subtask: false
---
Start the Furrow development loop with the goal from `$ARGUMENTS`.

## Usage
- `/furrow <goal>` — Start a new development loop for `<goal>`.
- `/furrow --continue` — Resume the last goal from `.kilo/furrow-state.md` (useful after a session was closed).
- `/furrow --status` — Print the current goal and remaining tasks without starting the loop.
- `/furrow <goal> --continue` — Override the stored goal and start fresh.

If no goal and no flag is provided, ask the user for one before starting.
Parses `$ARGUMENTS` for the leading flag; the remainder is the goal text.
