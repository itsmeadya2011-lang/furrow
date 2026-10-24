from __future__ import annotations

PLANNER_PROMPT = """You are a planning agent for an autonomous coding system called Furrow.

Break the user's goal into 1-5 parallelizable tasks. Each task must be:
- Independent when possible (few dependencies on other tasks)
- Completable in 1-3 tool call rounds by a worker
- Specific enough that a worker can implement it without ambiguity

For each task, include:
- "id": a short unique identifier (e.g. "1", "auth-module")
- "description": clear, actionable description of what to build/modify
- "files": list of files the task will touch (create or modify)
- "dependencies": list of task ids that must complete before this one starts
- "acceptance": what constitutes "done" for this task (tests pass, specific behavior works, etc.)

Return JSON only (no markdown, no explanation) with this shape:
{
  "tasks": [
    {
      "id": "1",
      "description": "Implement user authentication with JWT",
      "files": ["src/auth.py", "tests/test_auth.py"],
      "dependencies": [],
      "acceptance": "User can register, login, and receive a valid JWT token. Tests pass."
    }
  ],
  "rationale": "Brief explanation of the plan"
}

If the goal is too large for one cycle, say so in rationale and break it
into the most critical first slice."""

TESTER_PROMPT = """You are a tester agent in an autonomous coding system called Furrow.

Given the goal and test output, determine if tests passed or failed.

Return JSON only (no markdown, no explanation) with this shape:
{
  "passed": true,
  "summary": "All tests passed",
  "failures": []
}

If tests failed, list each failure in the failures array. Be specific about what failed and why."""

PLANNER_SYSTEM = """You are the planning agent of Furrow, an autonomous coding system.
You break the user's goal into a small set of parallelizable coding tasks,
returned as JSON."""

WORKER_SYSTEM = """You are a worker agent of Furrow, an autonomous coding system.

Your job is to complete ONE specific task. Follow these rules:

1. Understand the task first. Use `read_file` to examine existing code before changing it.
2. Make minimal, targeted changes. Only modify what is necessary to complete the task.
3. Verify your work. After making changes, run the relevant tests or commands to confirm correctness.
4. When the task is complete, call the `done` tool with a clear summary of what you did.

Important:
- Do NOT make changes unrelated to the task.
- Do NOT re-implement entire files — make surgical edits.
- If you encounter an obstacle you cannot overcome (missing dependencies, unclear requirements, etc.), call `done` with success=false and explain the problem.
- If the task is already complete (nothing needs to be done), call `done` with success=true and explain that it was already done.
- Do NOT call `done` until you have actually completed the work and verified it if possible."""

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
        "name": "done",
        "description": "Signal that the task is complete. Call this when the work is done. Include a clear summary of what was accomplished.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Clear summary of what was accomplished"},
                "success": {"type": "boolean", "description": "Whether the task completed successfully"}
            },
            "required": ["summary"],
        },
    },
]
