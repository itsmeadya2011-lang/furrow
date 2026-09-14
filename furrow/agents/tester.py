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
        completed = [t for t in tasks if t.status == "completed"]
        failed = [t for t in tasks if t.status == "failed"]
        pending = [t for t in tasks if t.status == "pending"]

        # If tasks are still pending, we cannot meaningfully test yet.
        if pending:
            return TestResult(
                passed=False,
                summary=f"{len(pending)} task(s) still pending; cannot test yet.",
                failures=[f"Task {t.id} is still pending" for t in pending],
            )

        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = (
            f"{TESTER_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Completed tasks: {len(completed)}\n"
            f"Failed tasks: {len(failed)}\n"
            f"Test output:\n{test_output}\n"
        )
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
        outputs: list[str] = []
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    continue
                exit_code = proc.returncode
                out = stdout.decode() + stderr.decode()
                outputs.append(f"Command: {' '.join(cmd)}\nExit code: {exit_code}\n{out}")
                if exit_code == 0 and out.strip():
                    return "\n---\n".join(outputs)
            except FileNotFoundError:
                continue
            except Exception:
                continue
        if outputs:
            return "\n---\n".join(outputs)
        return "No test runner found."
