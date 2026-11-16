from __future__ import annotations

PLANNER_PROMPT = """You are a planning agent for an autonomous coding system called Furrow.

Break the user's goal into 1-5 parallelizable tasks. Each task must be:
- Independent when possible
- Completable in 1-3 tool call rounds by a worker
- Specific enough that a worker can implement it without ambiguity

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

If the goal is too large for one cycle, say so in rationale and break it
into the most critical first slice."""

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
        "name": "git_status",
        "description": "show the working tree status (modified, staged, untracked files).",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "git_diff",
        "description": "show changes between commits, commit and working tree, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "staged": {"type": "boolean", "description": "show staged changes instead of unstaged"},
                "path": {"type": "string", "description": "limit diff to a specific file or directory"},
            },
            "required": [],
        },
    },
    {
        "name": "git_commit",
        "description": "commit staged changes with a message.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "commit message"},
                "add_all": {"type": "boolean", "description": "stage all modified and deleted files before commit"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "git_log",
        "description": "show recent commit history.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "maximum number of commits to show", "default": 10},
                "oneline": {"type": "boolean", "description": "show compact one-line format", "default": true},
            },
            "required": [],
        },
    },
]
