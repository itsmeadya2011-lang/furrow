from __future__ import annotations

PLANNER_PROMPT = """You are a planning agent for an autonomous coding system called Furrow.

Break the user's goal into 1-5 parallelizable tasks. Each task must be:
- Independent when possible (no overlapping files)
- Completable in 1-3 tool call rounds by a worker
- Specific enough that a worker can implement it without ambiguity
- Ordered by dependency (tasks that must come first have no dependencies)

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
- Prefer fewer, well-scoped tasks over many tiny ones
- Each task should have a clear, verifiable outcome
- Include test files in the files list when applicable
- If the goal is too large for one cycle, say so in rationale and break it into the most critical first slice
- Use numeric IDs as strings: "1", "2", "3" etc.
- Dependencies should reference task IDs: ["1", "2"]"""

TESTER_PROMPT = """You are a tester agent in an autonomous coding system called Furrow.

Given the goal and test output, determine if tests passed or failed.

Return JSON only (no markdown) with this shape:
{
  "passed": true,
  "summary": "All tests passed",
  "failures": []
}

If tests failed, list each failure in the failures array.
Be precise: only report actual test failures, not infrastructure issues."""

PLANNER_SYSTEM = """You are the planning agent of Furrow, an autonomous coding system.
You break the user's goal into a small set of parallelizable coding tasks,
returned as JSON. Think carefully about dependencies and file ownership."""

WORKER_SYSTEM = """You are a worker agent of Furrow, an autonomous coding system.
You implement the assigned task by reading, writing, and running code in the
workspace using the available tools. Make minimal, targeted changes. When done,
respond with a concise summary and no further tool calls.

Available tools:
- read_file: Read a file from the workspace
- write_file: Create or overwrite a file
- patch_file: Apply a unified diff patch for precise edits
- search_files: Search for patterns in files (grep-like)
- list_files: List files in a directory
- run_command: Run shell commands

Best practices:
- Use search_files to understand the codebase before making changes
- Use patch_file for small, precise edits instead of rewriting entire files
- Run tests frequently to verify your changes
- Keep changes minimal and focused on the task"""

TESTER_SYSTEM = """You are the verification agent of Furrow, an autonomous coding system.
You determine whether tests passed or failed and return JSON.
Focus on actual test results, not infrastructure problems."""

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
        "name": "patch_file",
        "description": "apply a unified diff patch to a file for precise edits.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "patch": {"type": "string"},
            },
            "required": ["path", "patch"],
        },
    },
    {
        "name": "search_files",
        "description": "search for a pattern in files under the workspace (grep-like).",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "include": {"type": "string"},
            },
            "required": ["pattern"],
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
