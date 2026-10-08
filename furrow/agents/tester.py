from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.agents.jsonutil import parse_json
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
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            return TestResult(**parse_json(response))
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    def _detect_test_commands(self) -> list[list[str]]:
        root = self.client.settings.workspace
        commands: list[list[str]] = []
        config_names = ("pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg")
        if any((root / name).exists() for name in config_names):
            commands.extend([["python", "-m", "pytest", "-q"], ["pytest", "-q"]])
        if (root / "package.json").exists():
            for runner in ("npm", "pnpm", "yarn"):
                commands.append([runner, "test", "--", "--silent"])
        if (root / "Cargo.toml").exists():
            commands.append(["cargo", "test", "-q"])
        if (root / "go.mod").exists():
            commands.append(["go", "test", "./..."])
        # Fallback: try common runners regardless of project files.
        commands.extend(
            [
                ["python", "-m", "pytest", "-q"],
                ["pytest", "-q"],
                ["npm", "test", "--", "--silent"],
                ["cargo", "test", "-q"],
                ["go", "test", "./..."],
            ]
        )
        # Deduplicate while preserving order.
        seen: set[tuple[str, ...]] = set()
        unique: list[list[str]] = []
        for cmd in commands:
            key = tuple(cmd)
            if key not in seen:
                seen.add(key)
                unique.append(cmd)
        return unique

    async def _run_tests(self) -> str:
        for cmd in self._detect_test_commands():
            proc: asyncio.subprocess.Process | None = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
                return f"Command: {' '.join(cmd)} (exit {proc.returncode})\n{output}"
            except asyncio.TimeoutError:
                if proc is not None:
                    proc.kill()
                    await proc.wait()
                continue
            except (FileNotFoundError, OSError):
                continue
        return "No test runner found."