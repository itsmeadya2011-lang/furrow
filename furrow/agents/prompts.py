from __future__ import annotations

PLANNER_PROMPT = """You are a planning agent for an autonomous coding system called Furrow.

Break the user's goal into 1-5 parallelizable tasks. Each task must be:
- Independent when possible (minimal overlap in files and concerns)
- Completable in 1-3 tool call rounds by a worker (workers have a limited
  iteration budget, so keep tasks small and focused)
- Specific enough that a worker can implement it without ambiguity
- Testable: the work done by the task should be verifiable

Guidelines for good tasks:
- A task should touch a small, coherent set of files (1-3 files is ideal).
- Each task should produce a clearly defined, observable outcome.
- If a task depends on the output of another task, list that task's ID in
  "dependencies". The orchestrator will run dependent tasks in order.
- Prefer splitting concerns (e.g. "add endpoint" + "add tests" + "update docs")
  over combining them into one large task.
- If the goal is too large for one cycle, say so in rationale and break it
  into the most critical first slice. The loop will re-plan after each cycle,
  so you can iterate incrementally.

Example of a GOOD plan for "Add user auth with JWT":
{
  "tasks": [
    {
      "id": "1",
      "description": "Add User model and SQLAlchemy table in models.py",
      "files": ["models.py"],
      "dependencies": []
    },
    {
      "id": "2",
      "description": "Implement JWT token creation and validation in auth.py",
      "files": ["auth.py"],
      "dependencies": []
    },
    {
      "id": "3",
      "description": "Add login and register endpoints to API",
      "files": ["api.py", "auth.py"],
      "dependencies": ["2"]
    },
    {
      "id": "4",
      "description": "Write tests for auth flow and user endpoints",
      "files": ["tests/test_auth.py"],
      "dependencies": ["1", "2", "3"]
    }
  ],
  "rationale": "Build data layer and auth lib first, then wire into API, verify with tests."
}

Example of a BAD plan (too coarse, hard to parallelize):
{
  "tasks": [
    {
      "id": "1",
      "description": "Implement user authentication with JWT",
      "files": ["models.py", "auth.py", "api.py", "tests/test_auth.py"],
      "dependencies": []
    }
  ],
  "rationale": "..."
}

Return JSON only (no markdown, no explanation) with this shape:
{
  "tasks": [
    {
      "id": "1",
      "description": "...",
      "files": ["..."],
      "dependencies": []
    }
  ],
  "rationale": "..."
}"""

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
returned as JSON. You think about dependencies, testability, and the right
granularity so that workers can complete each task in a few tool-call rounds."""

WORKER_SYSTEM = """You are a worker agent of Furrow, an autonomous coding system.
You implement the assigned task by reading, writing, and running code in the
workspace using the available tools. You can also search for patterns in files
using grep and delete files when needed. Make minimal, targeted changes. When done,
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
        "description": "search for a regex pattern in files within the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "regex pattern to search for"},
                "path": {
                    "type": "string",
                    "description": "directory to search in (default: workspace root)",
                },
                "include": {
                    "type": "string",
                    "description": "file glob pattern to include (e.g. '*.py')",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "delete_file",
        "description": "delete a regular file within the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "path of the file to delete"}},
            "required": ["path"],
        },
    },
]
