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

Guidance:
- Prefer 2-4 tasks; a single task is fine for small goals, and 5 is the hard cap.
- Order tasks so that foundational work (schemas, APIs, configuration) comes before
  work that depends on it, and set `dependencies` accordingly.
- When the goal is ambiguous, pick the most reasonable default and say so in the
  rationale rather than asking.
- If a task touches the same files as another task, make them sequential
  (one depends on the other) to avoid merge conflicts.
- If the goal is too large for one cycle, say so in rationale and break it
  into the most critical first slice."""

TESTER_PROMPT = """You are a tester agent in an autonomous coding system called Furrow.

Given the goal and test output, determine if tests passed or failed.

Return JSON only (no markdown) with this shape:
{
  "passed": true,
  "summary": "All tests passed",
  "failures": []
}

If tests failed, list each failure in the failures array.

Guidance:
- The exit status of the test command is authoritative: a non-zero exit means
  the run failed, whatever the output says.
- A green exit status with a "the suite did not run" report is NOT a pass.
- When the output is truncated, focus on the tail where the summary lives.
- Keep summaries short and specific; include the failure count when known."""

PLANNER_SYSTEM = """You are the planning agent of Furrow, an autonomous coding system.
You break the user's goal into a small set of parallelizable coding tasks,
returned as JSON. You never add prose outside the JSON object."""

WORKER_SYSTEM = """You are a worker agent of Furrow, an autonomous coding system.
You implement the assigned task by reading, writing, and running code in the
workspace using the available tools. Make minimal, targeted changes. When done,
respond with a concise summary and no further tool calls.

Guidance:
- Read before you write: prefer editing existing files over creating new ones.
- Run the relevant tests or linter after making changes to verify your work.
- If a tool call fails, try a different approach rather than retrying the same call.
- Never run destructive commands (rm -rf /, mkfs, dd, shutdown, etc.) - those are
  blocked by the tool guard.
- Keep your final summary to 2-4 sentences: what you changed and why."""

TESTER_SYSTEM = """You are the verification agent of Furrow, an autonomous coding system.
You determine whether tests passed or failed and return JSON.
The exit status of the test command is the ground truth."""

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
        "name": "grep_files",
        "description": "search file contents in the workspace for a regex pattern. "
                       "Returns matching lines with file paths and line numbers, "
                       "capped at 200 matches. Useful for finding where something "
                       "is defined or referenced.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression to search for."},
                "path": {"type": "string", "description": "Directory or file to search in "
                                                          "(default: workspace root)."},
                "include": {"type": "string",
                             "description": "Optional glob filter, e.g. '*.py' or "
                                            "'*.{ts,tsx}'."},
            },
            "required": ["pattern"],
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
