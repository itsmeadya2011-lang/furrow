from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

if TYPE_CHECKING:
    from furrow.config import Settings


DEFAULT_TIMEOUT = 120


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        return self._parse_test_result(response)

    @staticmethod
    def _extract_json(response: str) -> dict:
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        raise ValueError(f"Failed to extract JSON from LLM response.\nResponse: {response}")

    @staticmethod
    def _parse_test_result(response: str) -> TestResult:
        try:
            data = TesterAgent._extract_json(response)
            return TestResult(**data)
        except Exception:  # noqa: BLE001
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        cmd = self._detect_test_command()
        if cmd is None:
            return "No test runner found."

        timeout = getattr(self.client.settings, "test_timeout", DEFAULT_TIMEOUT)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except (FileNotFoundError, OSError):
            return "No test runner found."

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return stdout.decode() + stderr.decode()
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.wait()
            except ProcessLookupError:
                pass
            return "Test runner timed out."

    def _detect_test_command(self) -> list[str] | None:
        workspace = Path(self.client.settings.workspace)

        command = self._detect_from_pyproject(workspace)
        if command is not None and shutil.which(command[0]):
            return command

        command = self._detect_from_files(workspace)
        if command is not None:
            return command

        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        for cmd in candidates:
            if shutil.which(cmd[0]):
                return cmd
        return None

    @staticmethod
    def _detect_from_pyproject(workspace: Path) -> list[str] | None:
        if tomllib is None:
            return None

        pyproject = workspace / "pyproject.toml"
        if not pyproject.exists():
            return None

        try:
            with pyproject.open("rb") as f:
                data = tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError):
            return None

        tool = data.get("tool", {})
        project = data.get("project", {})

        if "pytest" in tool:
            return ["pytest", "-q"]

        scripts = project.get("scripts", {})
        if "test" in scripts and isinstance(scripts["test"], str):
            return scripts["test"].split()

        poetry_scripts = tool.get("poetry", {}).get("scripts", {})
        if "test" in poetry_scripts and isinstance(poetry_scripts["test"], str):
            return poetry_scripts["test"].split()

        if "coverage" in tool:
            return ["pytest", "-q"]

        return None

    @staticmethod
    def _detect_from_files(workspace: Path) -> list[str] | None:
        if workspace.joinpath("conftest.py").exists() or _has_py_tests(workspace):
            if shutil.which("pytest"):
                return ["pytest", "-q"]
            if shutil.which("python"):
                return ["python", "-m", "pytest", "-q"]

        if workspace.joinpath("package.json").exists():
            if shutil.which("npm"):
                return ["npm", "test", "--", "--silent"]
            if shutil.which("pnpm"):
                return ["pnpm", "test", "--", "--silent"]
            if shutil.which("yarn"):
                return ["yarn", "test", "--silent"]

        if workspace.joinpath("Cargo.toml").exists() and shutil.which("cargo"):
            return ["cargo", "test", "-q"]

        if workspace.joinpath("go.mod").exists() and shutil.which("go"):
            return ["go", "test", "./..."]

        return None


def _has_py_tests(workspace: Path) -> bool:
    patterns = ["test_*.py", "*_test.py", "tests/**/*.py", "test/**/*.py"]
    return any(list(workspace.glob(pattern)) for pattern in patterns)
