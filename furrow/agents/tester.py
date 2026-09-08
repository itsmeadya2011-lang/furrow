from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import TESTER_PROMPT, TESTER_FIX_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        log.info("tester.run.start", task_count=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            log.error("tester.run.test_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])

        if not result.passed and result.failures:
            try:
                log.info("tester.apply_fixes.start", failure_count=len(result.failures))
                await self.apply_fixes(goal, result.failures)
                new_output = await self._run_tests()
                interpret_prompt = (
                    f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{new_output}\n"
                )
                new_response = await self.client.complete(
                    interpret_prompt, model=self.client.settings.tester_model
                )
                try:
                    new_data = json.loads(new_response)
                    return TestResult(**new_data)
                except (json.JSONDecodeError, ValueError):
                    return TestResult(
                        passed="passed" in new_response.lower(),
                        summary=new_response,
                        failures=result.failures,
                    )
            except Exception as e:
                log.error("tester.apply_fixes.error", error=str(e))
                return TestResult(
                    passed=False,
                    summary=f"Fix attempt failed: {e}",
                    failures=result.failures,
                )

        return result

    async def apply_fixes(self, goal: str, failures: list[str]) -> str:
        file_paths = self._extract_file_paths(failures)
        if not file_paths:
            file_paths = self._discover_source_files()

        files_to_read = file_paths[:5]
        file_contents: dict[str, str] = {}
        for path in files_to_read:
            try:
                file_contents[path] = await self.client.read_file(path)
            except Exception:
                pass

        files_block = "\n\n".join(
            f"--- {path} ---\n{content}" for path, content in file_contents.items()
        )
        failures_block = "\n".join(failures)

        prompt = (
            f"{TESTER_FIX_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Failures:\n{failures_block}\n\n"
            f"Relevant files:\n{files_block}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return response

        summary = data.get("summary", "")
        for entry in data.get("files", []):
            path = entry.get("path", "")
            content = entry.get("content", "")
            if not path or not content:
                continue
            try:
                await self.client.write_file(path, content)
            except Exception:
                pass

        return summary

    def _extract_file_paths(self, failures: list[str]) -> list[str]:
        paths: list[str] = []
        pattern = re.compile(r'File "([^"]+)"')
        for failure in failures:
            matches = pattern.findall(failure)
            paths.extend(matches)
        return list(dict.fromkeys(paths))

    def _discover_source_files(self) -> list[str]:
        candidates = ["src", "lib", "furrow"]
        for directory in candidates:
            if Path(directory).is_dir():
                files = self.client.list_files(directory)
                return [f"{directory}/{f}" for f in files if not f.startswith("__") and not f.endswith(".pyc")]
        return []

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
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."
