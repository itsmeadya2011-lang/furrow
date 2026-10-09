---
description: Start the infinite Furrow development loop
agent: furrow
subtask: false
---
Start the Furrow development loop with the goal from $ARGUMENTS.

Usage: `/furrow <goal>` or `/furrow` to be prompted.

Options:
- `--model <model>` - Override LLM model (e.g., `--model gpt-4o`)
- `--provider <provider>` - LLM provider: anthropic, openai, ollama
- `--max-cycles <n>` - Maximum cycles (0 = unlimited)
- `--max-parallel <n>` - Maximum parallel tasks (default: 5)
- `--workspace <path>` - Workspace directory
- `--log-level <level>` - Log level: DEBUG, INFO, WARNING, ERROR
- `--log-file <path>` - Log file path

If no goal is provided, ask the user for one before starting.