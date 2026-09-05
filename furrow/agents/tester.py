from __future__ import annotations

import asyncio
import json

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult, TestRunOutput
from furrow.llm import LLMClient


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_run = await self._run_tests()
        test_output = self._format_test_output(test_run)
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            response = await self._retry_json(prompt, response)
            try:
                data = json.loads(response)
                return TestResult(**data)
            except (json.JSONDecodeError, ValueError):
                return TestResult(
                    passed="passed" in response.lower(),
                    summary=response,
                    failures=[],
                )

    async def _retry_json(self, prompt: str, original_response: str) -> str:
        retry_prompt = (
            f"{prompt}\n\n"
            f"Your previous response was not valid JSON. Previous response:\n{original_response}\n\n"
            "Respond ONLY with valid JSON matching this exact shape: "
            '{"passed": true, "summary": "...", "failures": []}. '
            "Do not include markdown or any other text."
        )
        return await self.client.complete(retry_prompt, model=self.client.settings.tester_model)

    def _format_test_output(self, run: TestRunOutput) -> str:
        parts = [
            f"Runner: {run.runner}",
            f"Command: {run.command}",
            f"Return code: {run.returncode}",
        ]
        if run.timed_out:
            parts.append("Status: timed out")
        if run.stdout:
            parts.append(f"Stdout:\n{run.stdout}")
        if run.stderr:
            parts.append(f"Stderr:\n{run.stderr}")
        return "\n".join(parts)

    async def _run_tests(self) -> TestRunOutput:
        candidates = self._detect_runners()
        timeout = self.client.settings.test_timeout
        for runner_name, cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    return TestRunOutput(
                        stdout=stdout.decode(),
                        stderr=stderr.decode(),
                        returncode=proc.returncode,
                        command=" ".join(cmd),
                        runner=runner_name,
                        timed_out=False,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    continue
            except OSError:
                continue
        return TestRunOutput(
            stdout="",
            stderr="",
            returncode=-1,
            command="",
            runner="none",
            timed_out=False,
        )

    def _detect_runners(self) -> list[tuple[str, list[str]]]:
        workspace = self.client.settings.workspace
        candidates: list[tuple[str, list[str]]] = []

        if (
            (workspace / "pyproject.toml").exists()
            or (workspace / "pytest.ini").exists()
            or (workspace / "setup.cfg").exists()
        ):
            candidates.append(("pytest", ["pytest", "-q"]))
            candidates.append(("pytest-module", ["python", "-m", "pytest", "-q"]))

        if (workspace / "package.json").exists():
            if (workspace / "node_modules" / ".bin" / "pnpm").exists():
                candidates.append(("pnpm", ["pnpm", "test", "--", "--silent"]))
            elif (workspace / "node_modules" / ".bin" / "yarn").exists():
                candidates.append(("yarn", ["yarn", "test", "--silent"]))
            else:
                candidates.append(("npm", ["npm", "test", "--", "--silent"]))

        if (workspace / "Cargo.toml").exists():
            candidates.append(("cargo", ["cargo", "test", "-q"]))

        if (workspace / "go.mod").exists():
            candidates.append(("go", ["go", "test", "./..."]))

        if not candidates:
            candidates.append(("pytest", ["pytest", "-q"]))
            candidates.append(("pytest-module", ["python", "-m", "pytest", "-q"]))

        return candidates
