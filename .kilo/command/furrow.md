---
description: Start the infinite Furrow development loop
agent: furrow
subtask: false
---
Start the Furrow development loop with the goal from $ARGUMENTS.
If no goal is provided, ask the user for one before starting.

## Usage

- `/furrow <goal>` — Start the development loop with the given goal.
- `/furrow --help` — Show a brief help message describing the command and its arguments.
- `/furrow` (no args) — Prompt for a goal inline before starting the loop.

## Behavior

The orchestrator runs continuously until the goal is complete or the loop is stopped. Mid-loop messages from the user are queued and incorporated into the next cycle's plan, so feedback is never lost. A persistent state file at `.kilo/furrow-state.json` tracks progress across cycles, making the loop resumable without requiring an explicit `/furrow resume` command.

## Examples

- `/furrow "Add JWT auth to the API"`
- `/furrow "Migrate the CLI to use commander.js"`
- `/furrow "Add unit tests for the payment service and make them green"`
