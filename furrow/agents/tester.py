from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


@dataclass
class TestRun:
    returncode: int
    stdout: str
    stderr: str
    runner: str
    timed_out: bool


def _extract_json(text: str) -> str | None:
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_run = await self._run_tests()
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest runner: {test_run.runner}\nExit code: {test_run.returncode}\nTimed out: {test_run.timed_out}\n\nStdout:\n{test_run.stdout}\n\nStderr:\n{test_run.stderr}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        result = self._try_parse_result(response, test_run)
        return result

    def _try_parse_result(self, response: str, test_run: TestRun) -> TestResult:
        json_text = _extract_json(response)
        if json_text is not None:
            try:
                data = json.loads(json_text)
                result = TestResult(**data)
            except (json.JSONDecodeError, ValueError):
                pass
            else:
                if test_run.timed_out or test_run.returncode != 0:
                    summary = f"Runner: {test_run.runner}, exit code: {test_run.returncode}"
                    if test_run.timed_out:
                        summary += " (timed out)"
                    failures = list(result.failures)
                    if test_run.stderr.strip():
                        failures.append(test_run.stderr.strip()[:500])
                    if test_run.stdout.strip():
                        failures.append(test_run.stdout.strip()[:500])
                    return TestResult(passed=False, summary=summary, failures=failures)
                return result

        passed = test_run.returncode == 0 and not test_run.timed_out
        summary = f"Runner: {test_run.runner}, exit code: {test_run.returncode}"
        if test_run.timed_out:
            summary += " (timed out)"
        failures = []
        if not passed:
            if test_run.stderr.strip():
                failures.append(test_run.stderr.strip()[:500])
            if test_run.stdout.strip():
                failures.append(test_run.stdout.strip()[:500])
        return TestResult(passed=passed, summary=summary, failures=failures)

    async def _run_tests(self) -> TestRun:
        workspace = self.client.settings.workspace
        runner = self._select_runner(workspace)
        if runner is None:
            return TestRun(returncode=127, stdout="", stderr="No test runner found", runner="none", timed_out=False)

        cmd, runner_name = runner
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=workspace, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                return TestRun(
                    returncode=proc.returncode,
                    stdout=stdout.decode(errors="replace"),
                    stderr=stderr.decode(errors="replace"),
                    runner=runner_name,
                    timed_out=False,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return TestRun(returncode=-1, stdout="", stderr="Test run timed out after 120s", runner=runner_name, timed_out=True)
        except FileNotFoundError:
            return TestRun(returncode=127, stdout="", stderr=f"Command not found: {cmd[0]}", runner=runner_name, timed_out=False)
        except Exception as e:
            return TestRun(returncode=1, stdout="", stderr=str(e), runner=runner_name, timed_out=False)

    def _select_runner(self, workspace: Path) -> tuple[list[str], str] | None:
        if self._has_pytest_config(workspace):
            return (["python", "-m", "pytest", "-q"], "pytest")
        if (workspace / "package.json").exists():
            return (["npm", "test", "--", "--silent"], "npm")
        if (workspace / "pnpm-lock.yaml").exists():
            return (["pnpm", "test", "--", "--silent"], "pnpm")
        if (workspace / "yarn.lock").exists():
            return (["yarn", "test", "--silent"], "yarn")
        if (workspace / "Cargo.toml").exists():
            return (["cargo", "test", "-q"], "cargo")
        if (workspace / "go.mod").exists():
            return (["go", "test", "./..."], "go")
        return None

    def _has_pytest_config(self, workspace: Path) -> bool:
        if (workspace / "pyproject.toml").exists():
            try:
                content = (workspace / "pyproject.toml").read_text()
                if re.search(r"^\[tool\.pytest(?:\.ini_options)?\]\s*$", content, re.MULTILINE):
                    return True
            except Exception:
                pass
        for config in ("pytest.ini", "setup.cfg", "tox.ini"):
            if (workspace / config).exists():
                return True
        for test_dir in ("tests", "test"):
            test_path = workspace / test_dir
            if test_path.exists() and (any(test_path.glob("test_*.py")) or any(test_path.glob("*_test.py"))):
                return True
        return False