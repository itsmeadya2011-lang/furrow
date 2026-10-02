from __future__ import annotations

import asyncio
import json

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

# Test runner configurations: (label, command list, timeout)
_TEST_RUNNERS: list[tuple[str, list[str]]] = [
    ("pytest", ["pytest", "-x", "--tb=short", "-q"]),
    ("python -m pytest", ["python", "-m", "pytest", "-x", "--tb=short", "-q"]),
    ("pytest-asyncio", ["pytest", "-x", "--tb=short", "-q", "-o", "asyncio_mode=auto"]),
    ("npm test", ["npm", "test", "--silent"]),
    ("pnpm test", ["pnpm", "test", "--silent"]),
    ("yarn test", ["yarn", "test", "--silent"]),
    ("cargo test", ["cargo", "test", "-q"]),
    ("go test", ["go", "test", "./..."]),
]


class TesterAgent:
    """Agent that runs tests and evaluates whether they pass."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run tests for the given goal and task list.

        Args:
            goal: The original (or updated) goal description.
            tasks: List of tasks that were executed in this cycle.
        """
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            return TestResult(
                passed=False,
                summary=f"Test execution error: {e}",
                failures=[f"Test runner error: {e}"],
            )

        if test_output.strip() == "No test runner found.":
            # No tests to run — check if tasks produced code changes
            return TestResult(
                passed=True,
                summary="No test runner found; assuming tasks completed successfully.",
                failures=[],
            )

        prompt = (
            f"{TESTER_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Tasks executed:\n"
            + ("\n".join(f"- {t.id}: {t.description}" for t in tasks) if tasks else "(no tasks)")
            + f"\n\nTest output:\n{test_output}\n"
            "Output JSON:\n"
        )

        # Always run heuristic evaluation on the test output to verify
        # the LLM's assessment against the actual test results.
        heuristic_result = self._heuristic_eval(test_output, "")

        try:
            data = await self.client.complete_json(
                prompt, model=self.client.settings.tester_model, temperature=0.3
            )
            llm_result = TestResult(
                passed=bool(data.get("passed", False)),
                summary=data.get("summary", ""),
                failures=data.get("failures", []),
            )
            # Use the heuristic's pass/fail determination since it reflects
            # the actual test output, while keeping the LLM's detailed summary
            # and failure messages.
            return TestResult(
                passed=heuristic_result.passed,
                summary=llm_result.summary or heuristic_result.summary,
                failures=llm_result.failures or heuristic_result.failures,
            )
        except (json.JSONDecodeError, ValueError):
            # Fallback: use heuristic result computed above
            return heuristic_result

    def _heuristic_eval(self, test_output: str, error: str) -> TestResult:
        """Fallback evaluation when LLM parsing fails."""
        output_lower = test_output.lower()
        if "passed" in output_lower and "failed" not in output_lower:
            passed = True
            summary = "Tests appear to have passed based on output analysis."
        elif "failed" in output_lower or "error" in output_lower:
            passed = False
            # Extract failure lines
            failures = [
                line.strip()
                for line in test_output.splitlines()
                if "fail" in line.lower() or "error" in line.lower()
            ]
            summary = f"Tests failed (LLM parse error: {error})"
            return TestResult(
                passed=passed, summary=summary, failures=failures or ["See test output for details"]
            )
        else:
            passed = False
            summary = f"Could not determine test results (LLM parse error: {error})"
            return TestResult(
                passed=passed, summary=summary, failures=["Unable to determine test status"]
            )

        return TestResult(passed=passed, summary=summary, failures=[])

    async def _run_tests(self) -> str:
        """Run the test suite, trying each configured test runner."""
        timeout = self.client.settings.test_timeout

        for label, cmd in _TEST_RUNNERS:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.client.settings.workspace),
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    output = stdout.decode(errors="replace")
                    err = stderr.decode(errors="replace")
                    # Return output if we got any meaningful result
                    if output.strip() or err.strip():
                        return output + ("\n" + err if err.strip() else "")
                except asyncio.TimeoutError:
                    proc.kill()
                    return f"[test runner '{label}' timed out after {timeout}s]"
            except FileNotFoundError:
                continue
            except OSError as e:
                return f"[test runner '{label}' error: {e}]"

        return "No test runner found."
