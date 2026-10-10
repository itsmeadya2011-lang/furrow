from __future__ import annotations

import asyncio
import json
import re
import shutil
from typing import TYPE_CHECKING, Any

import structlog

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

__all__ = ["TestResult", "TesterAgent"]

logger = structlog.get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("tester_agent.run.start", goal=goal, task_count=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            logger.error("tester_agent.run.error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        result = self._parse_test_result(response, test_output)
        logger.info("tester_agent.run.complete", passed=result.passed, summary=result.summary[:200])
        return result

    def _parse_test_result(self, response: str, test_output: str) -> TestResult:
        try:
            data = self._extract_json(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        passed = self._infer_passed_from_output(test_output)
        summary = response.strip() if response.strip() else test_output.strip()
        logger.warning(
            "tester_agent.parse.fallback", passed=passed, response_preview=response[:200]
        )
        return TestResult(passed=passed, summary=summary, failures=[])

    def _extract_json(self, text: str) -> dict[str, Any]:
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block_match:
            data: dict[str, Any] = json.loads(code_block_match.group(1))
            return data

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return data

        raise json.JSONDecodeError("No JSON found", text, 0)

    def _infer_passed_from_output(self, output: str) -> bool:
        if not output:
            return False

        output_lower = output.lower()

        # Check for explicit failure patterns (with numbers > 0)
        import re
        failed_match = re.search(r'(\d+)\s*failed', output_lower)
        if failed_match and int(failed_match.group(1)) > 0:
            return False
        
        error_match = re.search(r'(\d+)\s*error', output_lower)
        if error_match and int(error_match.group(1)) > 0:
            return False

        # Check for explicit FAILED/ERROR lines (common in pytest output)
        if re.search(r'\bfailed\b', output_lower) and not re.search(r'0\s*failed', output_lower):
            return False
        if re.search(r'\berror\b', output_lower) and not re.search(r'0\s*error', output_lower):
            return False

        failure_indicators = [
            "traceback",
            "exception",
            "assertionerror",
        ]

        success_indicators = [
            "passed",
            "ok",
            "success",
            "all passed",
            "tests passed",
        ]

        for indicator in failure_indicators:
            if indicator in output_lower:
                return False

        for indicator in success_indicators:
            if indicator in output_lower:
                return True

        return (
            "no test runner found" not in output_lower and "no tests collected" not in output_lower
        )

    async def _run_tests(self) -> str:
        sections = []

        pytest_cmds = [
            ["pytest", "-v"],
            ["python", "-m", "pytest", "-v"],
        ]

        for cmd in pytest_cmds:
            if await self._command_exists(cmd[0]):
                output, code = await self._run_command(cmd, "pytest")
                sections.append(f"=== pytest (exit code: {code}) ===\n{output}")
                break

        if shutil.which("ruff"):
            output, code = await self._run_command(["python", "-m", "ruff", "check", "."], "ruff")
            sections.append(f"=== ruff check (exit code: {code}) ===\n{output}")

        if shutil.which("mypy") or await self._command_exists("mypy"):
            output, code = await self._run_command(["python", "-m", "mypy", "."], "mypy")
            sections.append(f"=== mypy (exit code: {code}) ===\n{output}")

        npm_cmds = [
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
        ]

        for cmd in npm_cmds:
            if await self._command_exists(cmd[0]):
                output, code = await self._run_command(cmd, "npm test")
                sections.append(f"=== {cmd[0]} test (exit code: {code}) ===\n{output}")
                break

        other_cmds = [
            (["cargo", "test", "-q"], "cargo test"),
            (["go", "test", "./..."], "go test"),
        ]

        for cmd, name in other_cmds:
            if await self._command_exists(cmd[0]):
                output, code = await self._run_command(cmd, name)
                sections.append(f"=== {name} (exit code: {code}) ===\n{output}")
                break

        if not sections:
            return "No test runner found."

        return "\n\n".join(sections)

    async def _command_exists(self, cmd: str) -> bool:
        return shutil.which(cmd) is not None

    async def _run_command(self, cmd: list[str], name: str) -> tuple[str, int]:
        logger.info("tester_agent.run_command", command=cmd, name=name)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
                output = stdout.decode() + stderr.decode()
                logger.info(
                    "tester_agent.command_done",
                    name=name,
                    exit_code=proc.returncode,
                    output_len=len(output),
                )
                return output, proc.returncode if proc.returncode is not None else 1
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.warning("tester_agent.command_timeout", name=name)
                return "Timeout after 180s", 124
        except Exception as e:  # noqa: BLE001
            logger.error("tester_agent.command_error", name=name, error=str(e))
            return f"Error: {e}", 1

    async def run_tests_only(self) -> str:
        logger.info("tester_agent.run_tests_only.start")
        output = await self._run_tests()
        logger.info("tester_agent.run_tests_only.complete", output_len=len(output))
        return output