from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=f"Test execution failed: {e}", failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        try:
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        except Exception as e:
            return TestResult(passed=False, summary=f"LLM evaluation failed: {e}", failures=[str(e)])
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        timeout = self.client.settings.test_timeout
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        last_error = None
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    output = stdout.decode() + stderr.decode()
                    if proc.returncode == 0 or output.strip():
                        return output
                    last_error = f"Command {' '.join(cmd)} exited with code {proc.returncode} and no output"
                except asyncio.TimeoutError:
                    proc.kill()
                    try:
                        await proc.wait()
                    except Exception:
                        pass
                    last_error = f"Command {' '.join(cmd)} timed out after {timeout}s"
                    continue
            except FileNotFoundError:
                last_error = f"Command not found: {cmd[0]}"
                continue
            except PermissionError:
                last_error = f"Permission denied executing: {cmd[0]}"
                continue
            except OSError as e:
                last_error = f"OS error running {' '.join(cmd)}: {e}"
                continue
            except Exception as e:
                last_error = f"Unexpected error running {' '.join(cmd)}: {e}"
                continue
        return last_error or "No test runner found."
