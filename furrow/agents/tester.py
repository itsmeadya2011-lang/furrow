from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings


logger = get_logger("tester")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("testing_started", tasks_count=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            logger.error("test_run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # First, analyze test results
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            logger.warning("test_result_parse_failed", response=response[:200])
            result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])

        # If tests failed, try to fix them
        if not result.passed and result.failures:
            logger.info("attempting_test_fixes", failure_count=len(result.failures))
            await self._fix_failures(goal, test_output, result.failures)
            # Re-run tests after fixes
            try:
                test_output = await self._run_tests()
                prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output after fixes:\n{test_output}\n"
                response = await self.client.complete(prompt, model=self.client.settings.tester_model)
                try:
                    data = json.loads(response)
                    result = TestResult(**data)
                except (json.JSONDecodeError, ValueError):
                    result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])
            except Exception as e:  # noqa: BLE001
                logger.error("re_test_failed", error=str(e))

        logger.info("testing_completed", passed=result.passed, summary=result.summary)
        return result

    async def _fix_failures(self, goal: str, test_output: str, failures: list[str]) -> None:
        """Attempt to fix failing tests by asking LLM for fixes."""
        fix_prompt = f"""You are a tester agent fixing failing tests.

Goal: {goal}

Test output:
{test_output}

Failures:
{chr(10).join(f'  - {f}' for f in failures)}

Analyze the failures and provide fixes for the relevant files. Use the format:
```language
# filepath: path/to/file.py
<fixed code>
```

Focus only on fixing the test failures. Do not make unrelated changes."""
        
        response = await self.client.complete(fix_prompt, model=self.client.settings.tester_model)
        
        # Apply fixes
        pattern = r"```(\w+)?\n# filepath: (.+?)\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        
        for lang, file_path, content in matches:
            file_path = file_path.strip()
            content = content.strip()
            logger.info("applying_test_fix", file=file_path)
            try:
                await self.client.write_file(file_path, content)
            except OSError as e:
                logger.error("test_fix_write_failed", file=file_path, error=str(e))

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
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    logger.debug("test_command_output", cmd=cmd[0], exit_code=proc.returncode)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_command_timeout", cmd=cmd[0])
                    continue
            except (FileNotFoundError, OSError) as e:
                logger.debug("test_command_not_found", cmd=cmd[0], error=str(e))
                continue
        return "No test runner found."
