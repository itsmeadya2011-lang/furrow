from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from furrow.agents.parsing import extract_json_object, find_json_bool
from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)

_NO_TESTS_MARKERS = ("no tests ran", "no tests collected", "no test", "collected 0 items")


class TesterAgent:
    # Not a pytest test class: the name starts with "Test" but the class is a
    # production agent, not a collection target.
    __test__ = False

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            log.exception("run_tests_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = extract_json_object(response)
            return TestResult(**data)
        except Exception:
            passed_val = find_json_bool(response, "passed")
            if passed_val is not None:
                failures = []
                low = response.lower()
                if not passed_val:
                    for marker in ("failed", "failure", "traceback", "error", "assert"):
                        if marker in low:
                            failures.append(marker)
                return TestResult(passed=passed_val, summary=response, failures=failures)
            low = response.lower()
            if any(m in low for m in ("failed", "failure", "traceback", "error", "assert")):
                return TestResult(
                    passed=False,
                    summary=response,
                    failures=["failure marker present"],
                )
            return TestResult(passed=False, summary="could not determine test result", failures=[])

    async def _run_tests(self) -> str:
        cwd = Path.cwd()
        detected: list[list[str]] = []
        if (
            (cwd / "pyproject.toml").exists()
            or (cwd / "pytest.ini").exists()
            or (cwd / "tox.ini").exists()
            or (cwd / "tests").is_dir()
        ):
            detected.append(["pytest", "-q"])
            detected.append(["python", "-m", "pytest", "-q"])
        if (cwd / "package.json").exists():
            detected.append(["npm", "test", "--", "--silent"])
        if (cwd / "Cargo.toml").exists():
            detected.append(["cargo", "test", "-q"])
        if (cwd / "go.mod").exists():
            detected.append(["go", "test", "./..."])

        candidates: list[list[str]] = [
            *detected,
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]

        total_deadline = 120
        start = asyncio.get_event_loop().time()
        for cmd in candidates:
            if asyncio.get_event_loop().time() - start > total_deadline:
                break
            exe = cmd[0]
            if shutil.which(exe) is None:
                continue
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    continue
                out = (stdout.decode(errors="replace") + stderr.decode(errors="replace"))
                low = out.lower()
                if any(m in low for m in _NO_TESTS_MARKERS):
                    log.info("runner_no_tests", runner=exe)
                    continue
                log.info("runner_selected", runner=exe)
                return out
            except FileNotFoundError:
                continue
            except Exception as e:
                log.warning("runner_error", runner=exe, error=str(e))
                continue
        return "No test runner found."
