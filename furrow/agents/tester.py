from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT, extract_json_object
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

# Per-candidate wall-clock budget for a test command before it is killed.
TEST_TIMEOUT_SECONDS = 120
# How much of an unparseable model response is echoed back in the failure.
_RESPONSE_EXCERPT_CHARS = 200


class TesterAgent:
    # Not a pytest test class; prevent pytest from trying to collect it.
    __test__ = False

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001 - any runner failure is a failed test
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        task_lines = "\n".join(f"- [{task.id}] {task.description}" for task in tasks)
        task_block = task_lines or "(no tasks were planned for this cycle)"
        prompt = (
            f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTasks under test:\n{task_block}\n\n"
            f"Test output:\n{test_output}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = extract_json_object(response)
        except ValueError:
            return _unparseable_result(response)
        try:
            return TestResult(**data)
        except (TypeError, ValueError) as e:
            # Shape matched JSON but not TestResult; never guess `passed`.
            return TestResult(
                passed=False,
                summary=f"Tester JSON was not a valid result: {e}",
                failures=[f"Invalid tester response: {response[:_RESPONSE_EXCERPT_CHARS]}"],
            )

    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        # Run in the configured workspace, not the server's cwd. Tolerate stub
        # clients whose settings do not define a workspace.
        workspace = getattr(self.client.settings, "workspace", None) or None
        last_error: str | None = None

        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workspace,
                )
            except FileNotFoundError:
                # Runner binary is not installed; try the next candidate.
                continue
            except OSError as exc:
                last_error = f"{' '.join(cmd)}: {exc}"
                continue

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=TEST_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()  # reap the killed process before moving on
                return (
                    f"Test command timed out after {TEST_TIMEOUT_SECONDS}s: {' '.join(cmd)}"
                )
            except OSError as exc:
                last_error = f"{' '.join(cmd)}: {exc}"
                continue
            return stdout.decode(errors="replace") + stderr.decode(errors="replace")

        if last_error:
            return f"No test runner found. Last error: {last_error}"
        return "No test runner found."


def _unparseable_result(response: str) -> TestResult:
    """Build the failure result for a tester response that is not valid JSON."""
    excerpt = response[:_RESPONSE_EXCERPT_CHARS]
    return TestResult(
        passed=False,
        summary=f"Could not parse tester JSON response: {excerpt}",
        failures=[f"Unparseable tester response: {excerpt}"],
    )