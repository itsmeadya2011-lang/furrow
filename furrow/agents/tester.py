from __future__ import annotations

import asyncio
import json
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
        except Exception as e:  # noqa: BLE001 — tester must stay robust to any failure
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        output_parts: list[tuple[str, str]] = []

        # Run tests
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        test_output = None
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    test_output = stdout.decode() + stderr.decode()
                    break
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except Exception:  # noqa: BLE001, S112 — best-effort: try next runner
                continue
        if test_output is None:
            test_output = "No test runner found."
        output_parts.append(("TESTS", test_output))

        # Lint (ruff) — only if available
        try:
            proc = await asyncio.create_subprocess_exec(
                "ruff", "check", ".",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                output_parts.append(("LINT", stdout.decode() + stderr.decode()))
            except asyncio.TimeoutError:
                proc.kill()
        except Exception:  # noqa: BLE001, S110 — ruff not available; skip
            pass

        # Type check (mypy) — only if available
        try:
            proc = await asyncio.create_subprocess_exec(
                "mypy", "furrow/",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                output_parts.append(("TYPE CHECK", stdout.decode() + stderr.decode()))
            except asyncio.TimeoutError:
                proc.kill()
        except Exception:  # noqa: BLE001, S110 — mypy not available; skip
            pass

        return "\n\n".join(f"=== {name} ===\n{content}" for name, content in output_parts)
