from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = logging.getLogger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        log.info("tester.run_start", goal=goal, task_count=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            log.error("tester.run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            log.info("tester.result", passed=result.passed, summary=result.summary)
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            log.info("tester.result_parsed", passed=passed, response=response)
            return TestResult(passed=passed, summary=response, failures=[])

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
        output_parts = []
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    combined = stdout.decode() + stderr.decode()
                    output_parts.append(f"$ {' '.join(cmd)}\n{combined}")
                    if proc.returncode == 0:
                        return "\n".join(output_parts)
                except asyncio.TimeoutError:
                    proc.kill()
                    output_parts.append(f"$ {' '.join(cmd)}\n[timed out after 120s]")
                    continue
            except FileNotFoundError:
                continue
            except Exception as e:
                output_parts.append(f"$ {' '.join(cmd)}\n[error: {e}]")
                continue
        return "\n".join(output_parts) if output_parts else "No test runner found."
