from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from furrow.agents.planner import extract_json
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger()

NODE_LOCKFILES = (
    ("pnpm-lock.yaml", "pnpm"),
    ("yarn.lock", "yarn"),
    ("bun.lock", "bun"),
)

PYTHON_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", "pytest.ini", "tox.ini")

PYTEST_CANDIDATES = [["python", "-m", "pytest", "-q"], ["pytest", "-q"]]


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            log.warning("tests.result", passed=False, summary=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(extract_json(response))
            result = TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])
        log.info("tests.result", passed=result.passed, summary=result.summary)
        return result

    def _commands(self) -> list[list[str]]:
        """Pick a test command list based on which project markers exist."""
        workspace = Path(self.client.settings.workspace)

        if (workspace / "Cargo.toml").exists():
            return [["cargo", "test", "-q"]]
        if (workspace / "go.mod").exists():
            return [["go", "test", "./..."]]
        if (workspace / "package.json").exists():
            pm = "npm"
            for lockfile, name in NODE_LOCKFILES:
                if (workspace / lockfile).exists():
                    pm = name
                    break
            return [[pm, "test", "--", "--silent"]]
        if any((workspace / marker).exists() for marker in PYTHON_MARKERS):
            return [["python", "-m", "pytest", "-q"]]

        return [list(cmd) for cmd in PYTEST_CANDIDATES]

    async def _run_tests(self) -> str:
        timeout = getattr(self.client.settings, "test_timeout", 120)

        for cmd in self._commands():
            started = time.monotonic()
            log.debug("tests.start", command=cmd, timeout=timeout)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
            except FileNotFoundError:
                log.debug("tests.runner_missing", command=cmd)
                continue

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
                log.warning("tests.timeout", command=cmd, timeout=timeout)
                continue

            duration = round(time.monotonic() - started, 3)
            log.info("tests.finished", command=cmd, returncode=proc.returncode, duration=duration)
            return stdout.decode(errors="replace") + stderr.decode(errors="replace")

        log.debug("tests.no_runner")
        return "No test runner found."