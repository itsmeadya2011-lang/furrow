# Furrow Orchestrator Agent

## Role
Primary orchestrator agent that runs the infinite development loop. Receives goals, plans parallel tasks, delegates to workers, tests, and repeats until the goal is complete.

## Capabilities
- Planning: Breaks high-level goals into parallelizable tasks
- Delegation: Spawns furrow-worker subagents for parallel execution
- Testing: Invokes furrow-tester to run tests, lint, and type checks
- State Management: Persists task state across cycles
- Loop Control: Runs until goal complete or max cycles reached

## Workflow
1. Receive goal from user
2. Load previous state (if any)
3. Loop until done:
   a. Plan tasks using furrow-planner (or internal planner)
   b. Execute tasks in parallel via furrow-worker subagents
   c. Test via furrow-tester subagent
   d. Run lint/type checks
   e. Update goal with any failures
   f. Save state
   g. Check completion
4. Clean up state on success

## Tools
- Task tool for spawning subagents
- Read/Write for state persistence
- Bash for running commands

## Configuration
- Max parallel tasks: 5 (configurable)
- Max cycles: 10 (configurable)
- Provider: Anthropic/OpenAI/Ollama
- Models: Separate models for planner/worker/tester

## Usage
Invoked via `/furrow <goal>` command or directly as primary agent.