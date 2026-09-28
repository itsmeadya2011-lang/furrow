from __future__ import annotations

import asyncio
import json
import shutil
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger("furrow")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        if client is not None:
            self.client = client
        elif settings is not None:
            self.client = LLMClient(settings=settings)
        else:
            self.client = LLMClient()

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        task_list = "\n".join(f"- [{t.id}] {t.description}" for t in tasks)
        test_output = await self._run_tests()

        prompt = (
            f"{TESTER_PROMPT}\n\nGoal: {goal}\n\n"
            f"Tasks:\n{task_list}\n\n"
            f"Test output:\n{test_output}\n"
        )
        logger.info("tester_evaluate", output_len=len(test_output))
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            result = TestResult(
                passed="passed" in response.lower(),
                summary=response,
                failures=[],
            )
        logger.info("tester_done", passed=result.passed, summary=result.summary)
        return result

    async def _run_tests(self) -> str:
        cmd = self._detect_test_command()
        if cmd is None:
            return "No test runner found."
        logger.info("tester_running_tests", command=" ".join(cmd))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.client.settings.workspace),
            )
        except (FileNotFoundError, OSError) as e:
            logger.debug("tester_test_command_error", error=repr(e))
            return "No test runner found."
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return "Test command timed out after 120s"
        await proc.wait()
        transport = getattr(proc, "_transport", None)
        if transport is not None and not transport.is_closing():
            transport.close()
        return (stdout.decode(errors="replace") + stderr.decode(errors="replace")).strip()

    def _detect_test_command(self) -> list[str] | None:
        workspace = self.client.settings.workspace
        candidates: list[list[str]] = []
        if shutil.which("pytest") is not None:
            candidates.append(["pytest", "-q"])
        candidates.append(["python", "-m", "pytest", "-q"])
        if (workspace / "package.json").exists():
            pkg = self._detect_pkg_manager()
            if pkg:
                candidates.append([pkg, "test", "--silent"])
        if (workspace / "Cargo.toml").exists():
            candidates.append(["cargo", "test", "-q"])
        if (workspace / "go.mod").exists():
            candidates.append(["go", "test", "./..."])

        seen: set[tuple[str, ...]] = set()
        for cmd in candidates:
            key = tuple(cmd)
            if key in seen:
                continue
            seen.add(key)
            if shutil.which(cmd[0]) is not None:
                return cmd
        return None

    def _detect_pkg_manager(self) -> str | None:
        if shutil.which("pnpm") is not None:
            return "pnpm"
        if shutil.which("yarn") is not None:
            return "yarn"
        if shutil.which("npm") is not None:
            return "npm"
        return None
