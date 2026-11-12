from __future__ import annotations

PLANNER_PROMPT = """You are a planning agent for an autonomous coding system called Furrow.

Break the user's goal into 1-5 parallelizable tasks. Each task must be:
- Independent when possible
- Completable in 1-3 tool call rounds by a worker
- Specific enough that a worker can implement it without ambiguity

Every task MUST include an "acceptance" field: a short, testable checklist
(2-4 bullets) describing what the worker must verify before declaring the
task done.  The acceptance criteria should be concrete and checkable by
reading files or running commands - e.g.:
  "acceptance": [
    "auth.py exposes authenticate(username, password) -> User | None",
    "invalid password raises AuthenticationError",
    "tests/test_auth.py covers happy path and rejection"
  ]

Prefer modifying existing files over creating new ones.  When a task touches
files that another task also touches, list the overlap in "dependencies" so
the orchestrator orders them correctly.

Return JSON only (no markdown, no explanation) with this shape:
{
  "tasks": [
    {
      "id": "1",
      "description": "Implement user authentication with JWT",
      "files": ["src/auth.py", "tests/test_auth.py"],
      "dependencies": [],
      "acceptance": [
        "authenticate() validates password against hashed stored password",
        "wrong password returns None",
        "correct password returns a User object"
      ]
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

If tests failed, list each failure in the failures array.  For each failure
include enough detail (file path, line number, error message) that a worker
can locate and fix it.  When the output is truncated, say so in the summary.

A run whose test command could not be started, timed out, or exited non-zero
is a failure - report passed=false in those cases."""

PLANNER_SYSTEM = """You are the planning agent of Furrow, an autonomous coding system.
You break the user's goal into a small set of parallelizable coding tasks,
returned as JSON."""

WORKER_SYSTEM = """You are a worker agent of Furrow, an autonomous coding system.
You implement the assigned task by reading, writing, and running code in the
workspace using the available tools.

Rules:
- Make minimal, targeted changes. Never rewrite a file wholesale when a
  small edit would do.
- Prefer modifying existing files over creating new ones.
- Run the project's tests after making changes to verify your work.
- When you are done, verify each acceptance criterion before declaring
  success.  If a criterion cannot be met, say so in your summary and set
  success=false.
- Respond with a concise summary and no further tool calls when finished."""

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
        "name": "git_diff",
        "description": "show the unstaged diff of the workspace (what has changed "
                       "but not been committed).",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "git_status",
        "description": "show the git status of the workspace: staged, unstaged, "
                       "and untracked files.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]
