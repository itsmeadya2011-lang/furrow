from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow import logger

if TYPE_CHECKING:
    from furrow.config import Settings

log = logger.bind(component="tester")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            log.error("test_run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        log.debug("testing", goal=goal[:100], output_len=len(test_output))
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            passed = data.get("passed", False)
            if not passed:
                # Try to fix the failures
                log.info("tests_failed_attempting_fix", failures=data.get("failures", []))
                await self._fix_failures(data.get("failures", []), test_output)
                # Re-run tests after fix
                test_output = await self._run_tests()
                response = await self.client.complete(prompt, model=self.client.settings.tester_model)
                data = json.loads(response)
            result = TestResult(**data)
            log.info("test_result", passed=result.passed, summary=result.summary)
            return result
        except (json.JSONDecodeError, ValueError):
            log.warning("test_parse_failed", response=response[:200])
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

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
                    log.debug("test_run", cmd=cmd[0], returncode=proc.returncode, output_len=len(output))
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    log.warning("test_timeout", cmd=cmd[0])
                    continue
            except (FileNotFoundError, Exception) as e:
                log.debug("test_cmd_not_found", cmd=cmd[0], error=str(e))
                continue
        log.warning("no_test_runner_found")
        return "No test runner found."

    async def _fix_failures(self, failures: list[str], test_output: str) -> None:
        if not failures:
            return
        
        prompt = f"""You are a test fixer. The tests failed with these failures:
{chr(10).join(failures)}

Full test output:
{test_output}

Analyze the failures and fix the code. Return a JSON object with this shape:
{{
  "changes": [
    {{"file": "path/to/file", "content": "new file content"}}
  ],
  "summary": "What was fixed"
}}"""
        log.info("fixing_tests", num_failures=len(failures))
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    await self.client.write_file(file_path, content)
            log.info("test_fixes_applied", changes=len(changes))
        except (json.JSONDecodeError, ValueError):
            log.error("test_fix_parse_failed", response=response[:200])
