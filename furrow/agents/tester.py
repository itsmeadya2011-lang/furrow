from __future__ import annotations

import asyncio
import json
import subprocess

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings or Settings())

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Also read test files to understand what's being tested
        test_files_content = {}
        for task in tasks:
            for file_path in task.files:
                if "test" in file_path.lower() or "spec" in file_path.lower():
                    try:
                        content = await self.client.read_file(file_path)
                        test_files_content[file_path] = content
                    except Exception:  # noqa: BLE001, S110
                        pass

        test_files_context = ""
        if test_files_content:
            test_files_context = "\n\nTest file contents:\n"
            for path, content in test_files_content.items():
                test_files_context += f"\n--- {path} ---\n{content[:3000]}\n"

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}{test_files_context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
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
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, OSError, subprocess.SubprocessError):
                continue
        return "No test runner found."
