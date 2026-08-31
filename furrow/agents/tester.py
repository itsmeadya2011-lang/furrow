from __future__ import annotations

import asyncio
import json
import os
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
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        workspace = self.client.settings.workspace
        manifest_command = self._detect_test_command(workspace)
        if manifest_command is not None:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *manifest_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
                    output = stdout.decode() + stderr.decode()
                    if proc.returncode != 0:
                        output = f"Exit code: {proc.returncode}\n{output}"
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    return f"Test command {' '.join(manifest_command)} timed out after 180 seconds."
            except (FileNotFoundError, Exception) as e:
                return f"Test command {' '.join(manifest_command)} failed: {e}"

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
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
                    output = stdout.decode() + stderr.decode()
                    if proc.returncode != 0:
                        output = f"Exit code: {proc.returncode}\n{output}"
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."

    def _detect_test_command(self, workspace: str) -> list[str] | None:
        if not os.path.isdir(workspace):
            return None

        pyproject = os.path.join(workspace, "pyproject.toml")
        setup_py = os.path.join(workspace, "setup.py")
        if os.path.isfile(pyproject) or os.path.isfile(setup_py):
            return ["pytest", "-q"]

        package_json = os.path.join(workspace, "package.json")
        if os.path.isfile(package_json):
            try:
                with open(package_json, "r") as f:
                    data = json.load(f)
                scripts = data.get("scripts", {})
                if "test" in scripts:
                    return ["npm", "test", "--silent"]
            except (json.JSONDecodeError, OSError):
                pass
            return ["npm", "test", "--", "--silent"]

        cargo_toml = os.path.join(workspace, "Cargo.toml")
        if os.path.isfile(cargo_toml):
            return ["cargo", "test", "-q"]

        go_mod = os.path.join(workspace, "go.mod")
        if os.path.isfile(go_mod):
            return ["go", "test", "./..."]

        pom_xml = os.path.join(workspace, "pom.xml")
        if os.path.isfile(pom_xml):
            return ["mvn", "test", "-q"]

        build_gradle = os.path.join(workspace, "build.gradle")
        build_gradle_kts = os.path.join(workspace, "build.gradle.kts")
        if os.path.isfile(build_gradle) or os.path.isfile(build_gradle_kts):
            return ["gradle", "test"]

        return None
