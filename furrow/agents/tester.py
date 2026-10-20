from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anthropic import AnthropicError
from openai import OpenAIError

from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


FIX_PROMPT = """You are a debugging agent for the autonomous coding system Furrow.

The project failed its automated checks. Diagnose the root cause and write a patch
that fixes the failures.

Rules:
- Only touch the files that are needed to fix the failures.
- Keep the changes minimal and targeted; do not refactor unrelated code.
- Never weaken, skip or delete a test to make it pass.
- Return the full new content of every file you change.

Return JSON only (no markdown fences, no commentary) with this shape:
{
  "summary": "what you changed to fix the failures",
  "files": {"path/to/file.py": "full new file content"}
}
"""

MAX_FIX_ATTEMPTS = 2

CHECK_PASSED = "passed"
CHECK_FAILED = "failed"
CHECK_SKIPPED = "skipped"

DEFAULT_TIMEOUT = 120.0
TEST_TIMEOUT = 300.0
LINT_TIMEOUT = 120.0
TYPE_TIMEOUT = 180.0

MAX_OUTPUT_LINES = 40

#: Output markers that mean "the tool is not installed" rather than "the check failed".
MISSING_TOOL_MARKERS = ("no module named", "command not found", "not found")

#: Errors that may be raised by the model client while asking for a fix.
LLM_ERRORS = (AnthropicError, OpenAIError, OSError, RuntimeError, ValueError)


@dataclass(frozen=True)
class Check:
    """A single automated check (tests, lint or type checking)."""

    name: str
    command: tuple[str, ...]
    timeout: float


@dataclass(frozen=True)
class CheckResult:
    """The outcome of running a single check."""

    name: str
    status: str
    returncode: int
    output: str

    @property
    def failed(self) -> bool:
        return self.status == CHECK_FAILED


@dataclass
class Patch:
    """A set of full-file replacements produced by the model."""

    summary: str
    files: dict[str, str]


def classify(returncode: int, output: str) -> str:
    """Turn an exit code plus output into ``passed`` / ``failed`` / ``skipped``."""
    if returncode == 0:
        return CHECK_PASSED
    lowered = output.lower()
    if any(marker in lowered for marker in MISSING_TOOL_MARKERS):
        return CHECK_SKIPPED
    return CHECK_FAILED


def tail(text: str, max_lines: int = MAX_OUTPUT_LINES) -> str:
    """Return the last ``max_lines`` lines of ``text``."""
    lines = text.rstrip().splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def module_available(module: str) -> bool:
    """Return True when ``module`` can be imported by the current interpreter."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def tool_command(module: str, fallback: str, args: tuple[str, ...]) -> tuple[str, ...]:
    """Build the command for a tool.

    ``python -m <module>`` is preferred because it always uses the interpreter the
    agent runs with. Only when the module is not importable do we fall back to
    looking the executable up on ``PATH``.
    """
    if module_available(module):
        return (sys.executable, "-m", module, *args)
    executable = shutil.which(fallback)
    if executable is not None:
        return (executable, *args)
    # Neither is available: keep the module invocation so the run reports a
    # missing tool (which counts as "skipped", not as a failure).
    return (sys.executable, "-m", module, *args)


def strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence, if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    newline = stripped.find("\n")
    if newline == -1:
        return stripped
    body = stripped[newline + 1 :]
    end = body.rfind("```")
    if end != -1:
        body = body[:end]
    return body.strip()


def balanced_slice(text: str, start: int) -> str | None:
    """Return the JSON object starting at ``start`` or ``None`` if it is unbalanced."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object from a model response."""
    text = strip_code_fences(raw)
    index = text.find("{")
    while index != -1:
        candidate = balanced_slice(text, index)
        if candidate is not None:
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                result: dict[str, Any] = parsed
                return result
        index = text.find("{", index + 1)
    return None


def parse_patch(raw: str) -> Patch | None:
    """Parse a model response into a :class:`Patch`, or ``None`` when unusable."""
    data = extract_json_object(raw)
    if data is None:
        return None
    files = data.get("files")
    if not isinstance(files, dict):
        return None
    contents: dict[str, str] = {}
    for path, content in files.items():
        if isinstance(path, str) and isinstance(content, str) and path:
            contents[path] = content
    if not contents:
        return None
    summary = data.get("summary")
    return Patch(summary=summary if isinstance(summary, str) else "", files=contents)


class TesterAgent:
    """Runs the project checks (tests, lint, types) and tries to fix what fails."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        if client is not None:
            self.client = client
        elif settings is not None:
            self.client = LLMClient(settings=settings)
        else:
            self.client = LLMClient()
        self._fix_error: str | None = None
        self._fix_summary: str = ""

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run every check, try to fix failures and report a deterministic verdict."""
        results = await self._run_checks()
        attempts = 0
        while self._failure_count(results) and attempts < MAX_FIX_ATTEMPTS:
            attempts += 1
            patch = await self._request_fix(goal, tasks, results, attempts)
            if patch is None:
                break
            if await self._apply_patch(patch) == 0:
                break
            new_results = await self._run_checks()
            if self._failure_count(new_results) < self._failure_count(results):
                results = new_results
        failures = [self._format_failure(result) for result in results if result.failed]
        return TestResult(
            passed=not failures,
            summary=self._format_summary(results, attempts),
            failures=failures,
        )

    async def _run_checks(self) -> list[CheckResult]:
        """Run every check once and classify the results."""
        results: list[CheckResult] = []
        for check in self._build_checks():
            returncode, output = await self._run_command(list(check.command), timeout=check.timeout)
            results.append(
                CheckResult(
                    name=check.name,
                    status=classify(returncode, output),
                    returncode=returncode,
                    output=output,
                )
            )
        return results

    def _build_checks(self) -> list[Check]:
        """The checks the tester runs: unit tests, linting and type checking."""
        return [
            Check(
                name="tests",
                command=(sys.executable, "-m", "pytest", "-q"),
                timeout=TEST_TIMEOUT,
            ),
            Check(
                name="lint",
                command=tool_command("ruff", "ruff", ("check", ".")),
                timeout=LINT_TIMEOUT,
            ),
            Check(
                name="types",
                command=tool_command("mypy", "mypy", ("furrow",)),
                timeout=TYPE_TIMEOUT,
            ),
        ]

    async def _run_command(self, cmd: list[str], timeout: float = DEFAULT_TIMEOUT) -> tuple[int, str]:
        """Run ``cmd`` and return its exit code plus the combined stdout/stderr."""
        cwd = self._workspace()
        if not cwd.is_dir():
            return 1, f"Workspace directory does not exist: {cwd}"
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return 127, f"command not found: {cmd[0]}"
        except OSError as exc:
            return 126, f"could not run {cmd[0]}: {exc}"
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return 124, f"Timeout: command timed out after {timeout:.0f}s: {' '.join(cmd)}"
        output = (stdout or b"").decode(errors="replace")
        errors = (stderr or b"").decode(errors="replace")
        if errors:
            output = f"{output}\n{errors}" if output else errors
        code = process.returncode if process.returncode is not None else 1
        return code, output

    def _workspace(self) -> Path:
        """The directory the checks run in and the only directory patches may touch."""
        workspace = getattr(self.client.settings, "workspace", None)
        if isinstance(workspace, Path):
            return workspace
        if isinstance(workspace, str) and workspace:
            return Path(workspace)
        return Path.cwd()

    def _resolve_in_workspace(self, raw_path: str) -> Path | None:
        """Resolve ``raw_path`` inside the workspace, or ``None`` when it escapes."""
        workspace = self._workspace().resolve()
        candidate = Path(raw_path)
        try:
            if candidate.is_absolute():
                resolved = candidate.resolve()
            else:
                resolved = (workspace / candidate).resolve()
        except (OSError, ValueError):
            return None
        if resolved == workspace:
            return None
        try:
            resolved.relative_to(workspace)
        except ValueError:
            return None
        return resolved

    async def _request_fix(
        self,
        goal: str,
        tasks: list[TaskModel],
        results: list[CheckResult],
        attempt: int,
    ) -> Patch | None:
        """Ask the model for a patch, or return ``None`` when none could be used."""
        prompt = self._build_fix_prompt(goal, tasks, results, attempt)
        try:
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        except LLM_ERRORS as exc:
            self._fix_error = f"model call failed: {exc}"
            return None
        patch = parse_patch(response)
        if patch is None:
            self._fix_error = "could not parse a patch from the model response"
            return None
        self._fix_error = None
        self._fix_summary = patch.summary
        return patch

    async def _apply_patch(self, patch: Patch) -> int:
        """Write the patched files, skipping anything outside the workspace."""
        written = 0
        for raw_path, content in patch.files.items():
            target = self._resolve_in_workspace(raw_path)
            if target is None:
                continue
            await self.client.write_file(target, content)
            written += 1
        return written

    def _build_fix_prompt(
        self,
        goal: str,
        tasks: list[TaskModel],
        results: list[CheckResult],
        attempt: int,
    ) -> str:
        """Build the prompt asking the model for a fix."""
        sections = [FIX_PROMPT, f"\nGoal: {goal}"]
        if tasks:
            task_lines = "\n".join(f"- [{task.id}] {task.description}" for task in tasks)
            sections.append(f"\nTasks:\n{task_lines}")
        sections.append(f"\nFix attempt {attempt} of {MAX_FIX_ATTEMPTS}.")
        for result in results:
            if result.failed:
                sections.append(
                    f"\n--- {result.name} check failed (exit code {result.returncode}) ---\n"
                    f"{tail(result.output)}"
                )
        return "\n".join(sections)

    @staticmethod
    def _failure_count(results: list[CheckResult]) -> int:
        """How many checks failed."""
        return sum(1 for result in results if result.failed)

    @staticmethod
    def _format_failure(result: CheckResult) -> str:
        """Render a failed check as a single failure entry."""
        details = tail(result.output)
        header = f"{result.name} check failed (exit code {result.returncode})"
        return f"{header}:\n{details}" if details else header

    def _format_summary(self, results: list[CheckResult], attempts: int) -> str:
        """Render the overall summary of a run."""
        parts = ", ".join(f"{result.name}: {result.status}" for result in results)
        summary = f"Checks -> {parts}"
        if attempts:
            summary += f" (fix attempts: {attempts})"
        if self._fix_summary:
            summary += f" (fix: {self._fix_summary})"
        if self._fix_error:
            summary += f" (fix error: {self._fix_error})"
        return summary
