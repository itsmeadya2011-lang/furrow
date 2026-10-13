from __future__ import annotations

import asyncio
import shlex

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        if client is not None:
            self.client = client
            self.settings = getattr(client, "settings", settings or Settings())
        else:
            self.settings = settings or Settings()
            self.client = LLMClient(settings=self.settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except (RuntimeError, OSError, ValueError, asyncio.TimeoutError) as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        try:
            data = await self.client.complete_json(
                prompt, model=self.client.settings.tester_model
            )
            return TestResult(**data)
        except ValueError:
            response = await self.client.complete(
                prompt, model=self.client.settings.tester_model
            )
            return TestResult(
                passed="passed" in response.lower(), summary=response, failures=[]
            )

    async def _run_tests(self) -> str:
        test_cmd = getattr(self.client.settings, "test_command", None)
        if test_cmd:
            candidates = [shlex.split(str(test_cmd))]
        else:
            candidates = [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
                ["cargo", "test", "-q"],
                ["go", "test", "./..."],
            ]
        timeout = int(getattr(self.client.settings, "test_timeout", 120))
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    out = stdout.decode(errors="replace") + stderr.decode(errors="replace")
                    return f"exit code: {proc.returncode}\n" + out
                except asyncio.TimeoutError:
                    proc.kill()
                    try:
                        await proc.wait()
                    except (ProcessLookupError, OSError):
                        pass
                    continue
            except FileNotFoundError:
                continue
            except OSError:
                continue
        return "No test runner found."
