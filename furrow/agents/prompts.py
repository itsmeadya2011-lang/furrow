from __future__ import annotations

PLANNER_PROMPT = """You are a planning agent for an autonomous coding system called Furrow.

Break the user's goal into 1-5 parallelizable tasks. Each task must be:
- Independent when possible (minimize dependencies between tasks)
- Completable in 1-3 tool call rounds by a worker
- Specific enough that a worker can implement it without ambiguity
- Focused on a single logical unit of work

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
- Use numeric IDs (1, 2, 3...) for new tasks
- Reference existing task IDs for dependencies when continuing from previous cycles
- Prefer modifying existing tasks over creating new ones when the work is similar
- If the goal is too large for one cycle, say so in rationale and break it into the most critical first slice
- Each task should have a clear, testable outcome
- Include relevant test files in the files list when applicable
- Consider the project structure when suggesting file paths

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
returned as JSON. You consider existing tasks and their status to avoid
redundant work and build incrementally."""

WORKER_SYSTEM = """You are a worker agent of Furrow, an autonomous coding system.
You implement the assigned task by reading, writing, and running code in the
workspace using the available tools. Make minimal, targeted changes. When done,
respond with a concise summary and no further tool calls.

Available tools:
- read_file: Read a file from the workspace
- write_file: Create or overwrite a file
- patch_file: Apply a unified diff patch to a file (preferred for small changes)
- list_files: List files in a directory
- run_command: Run a shell command

Best practices:
- Use patch_file for small, targeted changes instead of rewriting entire files
- Read files before modifying them
- Run tests after making changes to verify correctness
- Keep changes minimal and focused on the task"""

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
        "name": "patch_file",
        "description": "apply a unified diff patch to a file. Use for small, targeted changes.",
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
