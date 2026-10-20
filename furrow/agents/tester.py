from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path

from furrow.agents.parsing import parse_test_result
from furrow.agents.prompts import TESTER_PROMPT, TESTER_SYSTEM
from furrow.config import Settings, TaskModel, TestResult
from furrow.config import settings as global_settings
from furrow.llm import LLMClient


class TesterAgent:
    __test__ = False   # not a pytest test class
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)
        self.workspace = Path(workspace) if workspace is not None else Path(self.settings.workspace)

    async def detect_command(self) -> list[str] | None:
        has_package = (self.workspace / "package.json").is_file()
        has_pyproject = (self.workspace / "pyproject.toml").is_file()
        has_tests_dir = (self.workspace / "tests").is_dir()

        if has_package and not (has_pyproject and has_tests_dir):
            if (self.workspace / "pnpm-lock.yaml").is_file():
                return ["pnpm", "test", "--silent"]
            if (self.workspace / "yarn.lock").is_file():
                return ["yarn", "test", "--silent"]
            return ["npm", "test", "--silent"]

        python_markers = [
            "pyproject.toml",
            "pytest.ini",
            "setup.cfg",
            "tox.ini",
            "requirements.txt",
        ]
        if any((self.workspace / m).is_file() for m in python_markers):
            return [sys.executable, "-m", "pytest", "-q"]

        if (self.workspace / "Cargo.toml").is_file():
            return ["cargo", "test", "-q"]
        if (self.workspace / "go.mod").is_file():
            return ["go", "test", "./..."]

        return None

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            cmd = await self.detect_command()
            if cmd is None:
                return TestResult(
                    passed=True,
                    summary="No test runner detected for this project.",
                    failures=[],
                )

            timeout = min(self.settings.request_timeout, 300)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(self.workspace),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    await proc.wait()
                    output = "Test command timed out."
                    returncode = 1
                    stdout = b""
                    stderr = output.encode()
                else:
                    returncode = proc.returncode if proc.returncode is not None else 0
            except Exception:
                returncode = 1
                stdout = b""
                stderr = b"Failed to run test command."

            combined = (
                stdout.decode("utf-8", errors="replace")
                + "\n"
                + stderr.decode("utf-8", errors="replace")
            )
            if len(combined) > 20000:
                combined = combined[:20000]

            prompt = TESTER_PROMPT + "\n\nGoal: " + goal + "\n\nTest output:\n" + combined

            try:
                response = await self.client.complete(
                    prompt, system=TESTER_SYSTEM, model=self.settings.tester_model
                )
                result = parse_test_result(response)
                return result
            except Exception:
                passed = returncode == 0 and not _has_failure_keywords(combined)
                failures = []
                seen = set()
                for line in combined.splitlines():
                    ls = line.strip()
                    if ls and _looks_like_failure(ls) and ls not in seen:
                        seen.add(ls)
                        failures.append(ls)
                        if len(failures) >= 10:
                            break
                return TestResult(
                    passed=passed,
                    summary=combined[:500],
                    failures=failures,
                )
        except Exception:
            return TestResult(passed=False, summary="Tester error", failures=[])


def _has_failure_keywords(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in ("fail", "error", "traceback", "assertion"))


def _looks_like_failure(line: str) -> bool:
    lower = line.lower()
    return any(k in lower for k in ("fail", "error", "traceback", "assertion"))
