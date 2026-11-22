"""Code Review Agent for Furrow.

The Code Review Agent provides an additional verification layer that reviews
the changes made by workers before tests are run. It can catch issues that
tests might miss, such as code style, security concerns, and architectural
consistency.
"""

from __future__ import annotations

from furrow.agents.parsing import parse_json
from furrow.config import CodeReviewResult, Settings
from furrow.config import settings as global_settings
from furrow.llm import LLMClient


class CodeReviewAgent:
    """Code review agent that reviews changes before tests are run."""

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)

    async def review(
        self,
        goal: str,
        files_changed: list[str],
        workspace: str,
    ) -> CodeReviewResult:
        """Review the changes made in the workspace.

        Args:
            goal: The original goal/task description
            files_changed: List of files that were modified
            workspace: Path to the workspace

        Returns:
            CodeReviewResult with review findings
        """
        # Read the changed files to review their content
        file_contents = {}
        for file_path in files_changed:
            try:
                from furrow.agents.worker.command_policy import resolve_in_workspace
                from pathlib import Path
                resolved = resolve_in_workspace(Path(workspace), file_path)
                if resolved and resolved.exists():
                    content = await self.client.read_file(resolved)
                    file_contents[file_path] = content
            except Exception:
                # If we can't read a file, skip it
                pass

        if not file_contents:
            return CodeReviewResult(
                passed=True,
                summary="No files to review",
                issues=[],
            )

        # Build the review prompt
        prompt = self._build_prompt(goal, file_contents)

        model = self.settings.reviewer_model
        response = await self.client.complete(prompt, system=CODE_REVIEW_SYSTEM, model=model)

        try:
            data = parse_json(response)
            if isinstance(data, dict):
                return CodeReviewResult(**data)
        except Exception:
            pass

        # Fallback: try to extract from text
        return self._parse_fallback(response, file_contents)

    def _build_prompt(self, goal: str, file_contents: dict[str, str]) -> str:
        """Build the review prompt with file contents."""
        parts = [
            CODE_REVIEW_PROMPT,
            f"\nGoal: {goal}",
            "\nFiles changed:",
        ]
        for path, content in file_contents.items():
            parts.append(f"\n--- {path} ---")
            parts.append(content)
        parts.append("\n\nReturn JSON only with the review result.")
        return "\n".join(parts)

    def _parse_fallback(self, text: str, file_contents: dict[str, str]) -> CodeReviewResult:
        """Fallback parsing when JSON parsing fails."""
        import re
        lower = text.lower()

        # Check for explicit pass/fail
        passed = True
        if re.search(r'\b(fail|issue|problem|error|concern)\b', lower):
            passed = False

        # Extract issues
        issues = []
        for line in text.splitlines():
            line = line.strip()
            if line and re.search(r'\b(issue|problem|error|warning|concern|suggest)\b', line, re.IGNORECASE):
                issues.append(line)

        return CodeReviewResult(
            passed=passed,
            summary=text[:500],
            issues=issues[:10],
        )


CODE_REVIEW_SYSTEM = """You are the code review agent of Furrow, an autonomous coding system.
You review code changes for correctness, security, style, and architectural consistency.
Return JSON only."""

CODE_REVIEW_PROMPT = """Review the following code changes for the given goal.

Check for:
1. Correctness - Does the code correctly implement the goal?
2. Security - Are there any security vulnerabilities (SQL injection, XSS, path traversal, etc.)?
3. Style - Does the code follow common conventions for the language?
4. Architecture - Is the code well-structured and maintainable?
5. Testing - Are there adequate tests for the changes?
6. Performance - Any obvious performance issues?

Return JSON only (no markdown) with this shape:
{
  "passed": true,
  "summary": "Brief summary of the review",
  "issues": ["List of specific issues found, if any"]
}

If no issues are found, return an empty issues array and passed=true.
"""


# Re-export for backward compatibility when we add it to config
from furrow.config import CodeReviewResult  # noqa: E402