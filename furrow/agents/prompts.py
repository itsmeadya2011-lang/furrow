from __future__ import annotations

from furrow.agents.custom_tools import get_custom_tools

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
workspace using the available tools. You have access to git tools for version
control: git_status, git_diff, git_log, git_branch, git_commit, and git_checkout.
Make minimal, targeted changes. When done, respond with a concise summary and no further tool calls."""

TESTER_SYSTEM = """You are the verification agent of Furrow, an autonomous coding system.
You determine whether tests passed or failed and return JSON."""


def get_worker_tools() -> list[dict]:
    """Get the list of worker tools including custom tools."""
    tools = [
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
                    "staged": {"type": "boolean", "description": "show staged changes only"},
                },
                "required": [],
            },
        },
        {
            "name": "git_log",
            "description": "show commit logs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_count": {"type": "integer", "description": "maximum number of commits to show"},
                },
                "required": [],
            },
        },
        {
            "name": "git_branch",
            "description": "show the current branch name.",
            "parameters": {
                "type": "object",
                "properties": {},
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
                    "add_all": {"type": "boolean", "description": "stage all changes before committing"},
                },
                "required": ["message"],
            },
        },
        {
            "name": "git_checkout",
            "description": "switch branches or create a new branch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "branch": {"type": "string", "description": "branch name"},
                    "create": {"type": "boolean", "description": "create new branch"},
                },
                "required": ["branch"],
            },
        },
    ]
    # Add custom tools
    tools.extend(get_custom_tools())
    return tools


# For backward compatibility
WORKER_TOOLS = get_worker_tools()
