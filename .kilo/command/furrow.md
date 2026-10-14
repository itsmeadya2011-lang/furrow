# /furrow Slash Command

## Description
Start the Furrow autonomous development loop with a goal.

## Usage
```
/furrow <goal>
```

## Examples
```
/furrow Add JWT authentication to the API
/furrow Refactor the user service to use dependency injection
/furrow Fix the failing tests in the payment module
```

## Behavior
1. Invokes the furrow orchestrator agent with the provided goal
2. Orchestrator runs the infinite loop until goal complete or max cycles
3. Progress is displayed in real-time
4. User can interrupt at any time

## Arguments
- goal (required): The high-level objective for Furrow to accomplish

## Permissions
Requires: bash: allow, edit: allow, read: allow