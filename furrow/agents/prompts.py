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
into the most critical first slice.

Rules:
- Use short, stable task ids ("1", "2", ...) that never change between cycles.
- List every file the task will create or modify. A task that touches no
  specific files may leave this list empty.
- Declare a dependency when a task reads or writes a file another task also
  touches, so dependent tasks run after their dependencies.
- Keep each task small: a single, well-defined change rather than a feature.
"""

TESTER_PROMPT = """You are a tester agent in an autonomous coding system called Furrow.

Given the goal and the raw test output, determine whether the tests passed.

Return JSON only (no markdown, no explanation) with this shape:
{
  "passed": true,
  "summary": "All tests passed",
  "failures": []
}

If tests failed, list each failure in the failures array as a short,
descriptive string (e.g. "tests/test_auth.py::test_login - AssertionError:
token missing"). When the output is ambiguous, default to passed=false and
explain the ambiguity in the summary so the loop can retry.
"""

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
]
