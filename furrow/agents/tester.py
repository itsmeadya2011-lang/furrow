from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _extract_json(text: str) -> str:
    m = _JSON_CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_failures(output: str) -> list[str]:
    failures = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if any(marker in upper for marker in ("FAILED", "ERROR", "ASSERT", "TRACEBACK")):
            failures.append(stripped)
    return failures


class TesterAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        from furrow.config import settings as default_settings

        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)

        json_str = _extract_json(response)
        try:
            data = json.loads(json_str)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            pass

        passed = "passed" in response.lower() and "failed" not in response.lower()
        failures = _extract_failures(test_output)
        return TestResult(
            passed=passed,
            summary=response[:500],
            failures=failures,
        )

    async def _run_tests(self) -> str:
        from furrow.config import settings as default_settings

        settings = self.settings or default_settings

        candidates = []
        if settings.test_commands:
            candidates.append(list(settings.test_commands))

        candidates.extend(
            [
                ["pytest", "-q"],
                ["python", "-m", "pytest", "-q"],
                ["npm", "test", "--", "--silent"],
                ["pnpm", "test", "--", "--silent"],
                ["yarn", "test", "--silent"],
                ["cargo", "test", "-q"],
                ["go", "test", "./..."],
            ]
        )

        timeout = getattr(settings, "test_timeout", 120)
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    continue
            except (FileNotFoundError, Exception):  # noqa: BLE001
                continue

        return "No test runner found or no output produced."
