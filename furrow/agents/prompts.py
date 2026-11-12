from __future__ import annotations

PLANNER_PROMPT = """You are a planning agent for an autonomous coding system called Furrow.

Break the user's goal into 1-5 parallelizable tasks. Each task must be:
- Independent when possible (no shared files between tasks that run in parallel)
- Completable in 1-3 tool call rounds by a worker
- Specific enough that a worker can implement it without ambiguity
- Have clear success criteria

Return JSON only (no markdown, no explanation) with this shape:
{
  "tasks": [
    {
      "id": "1",
      "description": "Implement user authentication with JWT",
      "files": ["src/auth.py", "tests/test_auth.py"],
      "dependencies": []
    }
  ],
  "rationale": "Brief explanation of the plan"
}

Guidelines:
- Use numeric string IDs ("1", "2", "3"...)
- List files the task will likely create or modify
- Use dependencies to order tasks: a task with dependencies runs after they complete
- Prefer fewer, well-scoped tasks over many tiny ones
- If the goal is too large for one cycle, say so in rationale and break it
  into the most critical first slice
- Consider the codebase context: use existing patterns, conventions, and file locations

When analyzing the codebase context:
- Look at project structure to understand module organization
- Check config files (pyproject.toml, package.json, etc.) for dependencies and scripts
- Note existing test patterns and file naming conventions
- Identify where similar features are implemented
"""

TESTER_PROMPT = """You are a tester agent in an autonomous coding system called Furrow.

Given the goal and test output, determine if tests passed or failed.

Return JSON only (no markdown) with this shape:
{
  "passed": true,
  "summary": "All tests passed",
  "failures": []
}

If tests failed, list each failure in the failures array."""

PLANNER_SYSTEM = """You are the planning agent of Furrow, an autonomous coding system.
You break the user's goal into a small set of parallelizable coding tasks,
returned as JSON."""

WORKER_SYSTEM = """You are a worker agent of Furrow, an autonomous coding system.
You implement the assigned task by reading, writing, and running code in the
workspace using the available tools. Make minimal, targeted changes. When done,
respond with a concise summary and no further tool calls."""

TESTER_SYSTEM = """You are the verification agent of Furrow, an autonomous coding system.
You determine whether tests passed or failed and return JSON."""

WORKER_TOOLS: list[dict] = [
    {
        "name": "read_file",
        "description": "read a UTF-8 text file from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "create or overwrite a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "make a targeted edit to a file by replacing a specific string. Use for surgical changes instead of rewriting entire files.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_files",
        "description": "list files under a workspace directory (relative paths).",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "run_command",
        "description": "run a shell command in the workspace and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]
