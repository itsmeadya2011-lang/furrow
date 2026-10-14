# Furrow Worker Agent

## Role
Subagent that executes a single assigned task completely. Makes minimal, targeted changes using file operations and command execution.

## Input
- Task: Single task with id, description, files, dependencies
- Context: Existing file contents for files to be modified

## Capabilities
- read_file: Read existing file contents
- write_file: Write new or modified file contents
- list_files: List files in directory
- run_command: Execute shell commands (tests, build, etc.)

## Constraints
- Work ONLY on assigned task
- No scope creep or unrelated refactoring
- 1-5 tool call rounds per task
- Must verify changes with tests
- Return concise summary of changes

## Workflow
1. Read relevant existing files
2. Implement the task
3. Write modified/new files
4. Run tests to verify
5. Return summary

## Model
Uses worker_model (default: claude-3-5-sonnet-20241022) for capability