---
name: furrow
description: Start the Furrow infinite development loop for a given goal
---

# /furrow Slash Command

Invokes the `furrow` orchestrator agent to start an infinite development loop for the provided goal.

## Usage

```
/furrow <goal>
```

## Behavior

1. Receives the goal from `$ARGUMENTS`
2. Starts the `furrow` orchestrator agent with the goal
3. The orchestrator runs an infinite loop:
   - Plans parallel tasks via `furrow-planner`
   - Executes tasks via parallel `furrow-worker` subagents
   - Tests via `furrow-tester` subagent
   - Reports progress and repeats until goal is complete
4. The loop continues until the goal is complete or the user stops the session

## Permissions

The orchestrator runs with open permissions by default:
- `bash: allow`
- `edit: allow`
- `read: allow`

## Example

```
/furrow Build a REST API with authentication and rate limiting
/furrow Refactor the authentication module to use JWT tokens
/furrow Add comprehensive test coverage for the payment service
```