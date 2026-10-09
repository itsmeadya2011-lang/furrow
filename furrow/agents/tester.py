from __future__ import annotations

import asyncio
import json

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

log = get_logger("furrow.tester")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        lint_output = ""
        type_output = ""

        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            test_output = str(e)

        try:
            lint_output = await self._run_lint()
        except Exception as e:  # noqa: BLE001
            lint_output = str(e)

        try:
            type_output = await self._run_type_check()
        except Exception as e:  # noqa: BLE001
            type_output = str(e)

        combined_output = f"=== TESTS ===\n{test_output}\n\n=== LINT ===\n{lint_output}\n\n=== TYPE CHECK ===\n{type_output}"

        log.debug("testing", goal=goal[:100], tasks=len(tasks))
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest/Lint/Type output:\n{combined_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            log.info("testing_completed", passed=result.passed, summary=result.summary[:100], failures=len(result.failures))
            return result
        except (json.JSONDecodeError, ValueError):
            # Fallback: check if "passed" is in response
            passed = "passed" in response.lower() and "failed" not in response.lower()
            log.warning("tester_parse_failed", passed=passed, response=response[:200])
            return TestResult(
                passed=passed,
                summary=response[:500],
                failures=[] if passed else [response[:500]],
            )

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
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=self.settings.workspace
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if output.strip() or proc.returncode != 0:
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
            except Exception as e:  # noqa: BLE001
                log.warning("Command failed: %s", e)
                continue
        return "No test runner found or no tests to run."

    async def _run_lint(self) -> str:
        candidates = [
            ["ruff", "check", "."],
            ["flake8", "."],
            ["eslint", "."],
            ["golangci-lint", "run"],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=self.settings.workspace
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = stdout.decode() + stderr.decode()
                    if output.strip() or proc.returncode != 0:
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
            except Exception as e:  # noqa: BLE001
                log.warning("Command failed: %s", e)
                continue
        return "No linter found or no lint issues."

    async def _run_type_check(self) -> str:
        candidates = [
            ["mypy", "."],
            ["pyright", "."],
            ["tsc", "--noEmit"],
            ["go", "vet", "./..."],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=self.settings.workspace
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if output.strip() or proc.returncode != 0:
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
            except Exception as e:  # noqa: BLE001
                log.warning("Command failed: %s", e)
                continue