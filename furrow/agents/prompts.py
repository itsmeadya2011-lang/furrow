from __future__ import annotations

PLANNER_PROMPT = """You are a planning agent for an autonomous coding system called Furrow.

Break the user's goal into 1-5 parallelizable tasks. Each task must be:
- Independent when possible (minimize dependencies between tasks)
- Completable in 1-3 tool call rounds by a worker
- Specific enough that a worker can implement it without ambiguity
- Focused on a single logical change

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

Task IDs must be unique strings. Dependencies should reference other task IDs.
Files should list the files each task will likely modify.

If the goal is too large for one cycle, say so in rationale and break it
into the most critical first slice.

Planning guidelines:
- Prefer small, focused tasks over large monolithic ones
- Group related changes into the same task
- Order tasks by dependency (independent tasks first)
- Consider test files as part of the task
- Each task should be verifiable (tests should pass after completion)"""

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
    {
        "name": "grep",
        "description": (
            "search for a pattern in files within the workspace. "
            "Returns matching lines with file paths and line numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Directory or file to search in "
                        "(default: workspace root)"
                    )
                },
                "include": {
                    "type": "string",
                    "description": (
                        "File pattern to include "
                        "(e.g., '*.py', '*.{js,ts}')"
                    )
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "diff",
        "description": (
            "show a unified diff between two file versions or between "
            "a file and new content. Useful for previewing changes before applying them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to diff"
                },
                "new_content": {
                    "type": "string",
                    "description": "New content to compare against the current file"
                },
            },
            "required": ["path", "new_content"],
        },
    },
    {
        "name": "patch",
        "description": (
            "apply a unified diff patch to a file. The patch should be in "
            "standard unified diff format."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to patch"},
                "patch": {
                    "type": "string",
                    "description": "Unified diff patch content"
                },
            },
            "required": ["path", "patch"],
        },
    },
]
