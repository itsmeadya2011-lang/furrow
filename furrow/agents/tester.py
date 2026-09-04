from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import structlog

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)


class TestRunResult(TypedDict):
    command: str
    returncode: int
    stdout: str
    stderr: str


def _has_test_target(makefile: Path) -> bool:
    """Check whether a Makefile exposes a ``test`` target."""
    try:
        text = makefile.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return bool(re.search(r"^test\s*:", text, re.MULTILINE))


def _detect_test_command(workspace: Path) -> list[str]:
    """Detect the appropriate test command for the project in *workspace*."""
    if (workspace / "Cargo.toml").is_file():
        return ["cargo", "test"]
    if (workspace / "go.mod").is_file():
        return ["go", "test", "./..."]
    if (workspace / "Makefile").is_file() and _has_test_target(workspace / "Makefile"):
        return ["make", "test"]
    if (workspace / "package.json").is_file():
        if shutil.which("pnpm"):
            return ["pnpm", "test", "--silent"]
        if shutil.which("yarn"):
            return ["yarn", "test", "--silent"]
        if shutil.which("npm"):
            return ["npm", "test", "--", "--silent"]
    if any(
        (workspace / name).is_file()
        for name in ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", "setup.py")
    ):
        return ["pytest", "-q"]
    if shutil.which("pytest"):
        return ["pytest", "-q"]
    if shutil.which("python"):
        return ["python", "-m", "pytest", "-q"]
    return []


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        result = await self._run_tests()
        log.info(
            "tester.tests_executed",
            command=result["command"],
            returncode=result["returncode"],
            stdout_len=len(result["stdout"]),
            stderr_len=len(result["stderr"]),
        )

        test_context = (
            f"Command: {result['command']}\n"
            f"Returncode: {result['returncode']}\n"
            f"stdout:\n{result['stdout']}\n"
            f"stderr:\n{result['stderr']}\n"
        )
        prompt = (
            f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nStructured test results:\n{test_context}\n"
        )
        response = await self.client.complete(
            prompt, model=self.client.settings.tester_model
        )
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("tester.parse_failed", error=str(e), response_preview=response[:500])
            retry_prompt = (
                f"{prompt}\n\nPlease return only valid JSON with keys passed, summary, failures."
            )
            response = await self.client.complete(
                retry_prompt, model=self.client.settings.tester_model
            )
            try:
                data = json.loads(response)
                return TestResult(**data)
            except (json.JSONDecodeError, ValueError):
                log.error("tester.parse_failed_retry", response_preview=response[:500])
                return TestResult(
                    passed="passed" in response.lower(),
                    summary=response,
                    failures=[],
                )

    async def _run_tests(self) -> TestRunResult:
        workspace = self.client.settings.workspace
        timeout = self.client.settings.request_timeout
        command = _detect_test_command(workspace)
        if not command:
            log.warning("tester.no_test_command", workspace=str(workspace))
            return {
                "command": "",
                "returncode": -1,
                "stdout": "",
                "stderr": "No test runner found.",
            }

        log.debug("tester.running", command=command, timeout=timeout)
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(workspace),
            )
        except (FileNotFoundError, OSError) as e:
            log.warning("tester.exec_error", command=command, error=str(e))
            return {
                "command": " ".join(command),
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
            }

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            log.warning("tester.timeout", command=command, timeout=timeout)
            proc.kill()
            await proc.communicate()
            return {
                "command": " ".join(command),
                "returncode": -1,
                "stdout": "",
                "stderr": f"Test execution timed out after {timeout} seconds.",
            }

        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        returncode = proc.returncode if proc.returncode is not None else -1

        passed = returncode == 0
        log.info(
            "tester.results",
            command=" ".join(command),
            returncode=returncode,
            passed=passed,
        )
        return {
            "command": " ".join(command),
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
