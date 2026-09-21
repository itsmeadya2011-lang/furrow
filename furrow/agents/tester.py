from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult, settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from asyncio.subprocess import Process

@dataclass
class TestExecution:
    returncode: int | None
    output: str
    timed_out: bool = False
    command: tuple[str, ...] | None = None
    error: str | None = None

    def __str__(self) -> str:
        return self.output


class TesterAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | str | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.workspace = Path(workspace) if workspace is not None else Path(self.settings.workspace)
        self.test_timeout = float(getattr(self.settings, "test_timeout", 120))

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            execution = await self._run_tests()
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            execution = TestExecution(returncode=None, output=message, error=message)
        execution = self._coerce_execution(execution)
        summary, failures = await self._summarize(goal, execution)
        return TestResult(
            passed=execution.returncode == 0,
            summary=summary,
            failures=failures,
        )

    async def _summarize(self, goal: str, execution: TestExecution) -> tuple[str, list[str]]:
        fallback = self._fallback_summary(execution)
        prompt = (
            f"{TESTER_PROMPT}\n\nGoal: {goal}\n\n"
            f"Subprocess exit code: {execution.returncode}\n"
            f"Timed out: {'yes' if execution.timed_out else 'no'}\n"
            f"Test output:\n{execution.output}\n"
        )
        try:
            model = getattr(getattr(self.client, "settings", None), "tester_model", None)
            if model:
                response = await self.client.complete(prompt, model=model)
            else:
                response = await self.client.complete(prompt)
        except TypeError:
            try:
                response = await self.client.complete(prompt)
            except Exception:
                return fallback, self._fallback_failures(execution)
        except Exception:
            return fallback, self._fallback_failures(execution)

        try:
            data = json.loads(response)
            if not isinstance(data, dict):
                return fallback, self._fallback_failures(execution)
            summary = data.get("summary") or fallback
            raw_failures = data.get("failures", [])
            if isinstance(raw_failures, str):
                failures = [raw_failures] if raw_failures else []
            elif isinstance(raw_failures, list):
                failures = [str(failure) for failure in raw_failures if str(failure)]
            else:
                failures = []
            if execution.returncode == 0:
                return str(summary), []
            if not failures:
                failures = self._fallback_failures(execution)
            return str(summary), failures
        except (json.JSONDecodeError, TypeError, ValueError):
            summary = response.strip() if isinstance(response, str) and response.strip() else fallback
            return summary, self._fallback_failures(execution) if execution.returncode not in (0, None) else []

    async def _run_tests(self) -> TestExecution:
        if not self.workspace.is_dir():
            message = f"Workspace does not exist or is not a directory: {self.workspace}"
            return TestExecution(returncode=None, output=message, error=message)

        candidates = (
            ("pytest", "-q"),
            ("python", "-m", "pytest", "-q"),
            ("npm", "test", "--", "--silent"),
            ("pnpm", "test", "--", "--silent"),
            ("yarn", "test", "--silent"),
            ("cargo", "test", "-q"),
            ("go", "test", "./..."),
        )
        missing_runners: list[str] = []
        for command in candidates:
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=self.workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                missing_runners.append(" ".join(command))
                continue
            except OSError as exc:
                message = f"Unable to start test runner {' '.join(command)}: {exc}"
                return TestExecution(returncode=None, output=message, command=command, error=str(exc))

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.test_timeout
                )
            except asyncio.TimeoutError:
                await self._stop_process(process)
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
                except Exception:
                    stdout = b""
                    stderr = b""
                output = self._decode(stdout) + self._decode(stderr)
                message = (
                    f"Test command timed out after {self.test_timeout:g} seconds: "
                    f"{' '.join(command)}"
                )
                return TestExecution(
                    returncode=124,
                    output=output or message,
                    timed_out=True,
                    command=command,
                    error=message,
                )
            except asyncio.CancelledError:
                await self._stop_process(process)
                raise

            output = self._decode(stdout) + self._decode(stderr)
            return TestExecution(
                returncode=process.returncode,
                output=output,
                command=command,
            )

        detail = f"Missing test runners: {', '.join(missing_runners)}" if missing_runners else ""
        message = "No test runner found."
        if detail:
            message = f"{message} {detail}"
        return TestExecution(returncode=None, output=message, error=detail or message)

    async def _stop_process(self, process: Process) -> None:
        if getattr(process, "returncode", None) is not None:
            return
        try:
            process.kill()
        except (AttributeError, ProcessLookupError, OSError):
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except (asyncio.TimeoutError, AttributeError, ProcessLookupError, OSError):
            return

    @staticmethod
    def _decode(value: bytes | str) -> str:
        if isinstance(value, str):
            return value
        return value.decode(errors="replace")

    @staticmethod
    def _coerce_execution(value: Any) -> TestExecution:
        if isinstance(value, TestExecution):
            return value
        if isinstance(value, tuple) and len(value) >= 2:
            return TestExecution(
                returncode=value[0],
                output=str(value[1]),
                timed_out=bool(value[2]) if len(value) > 2 else False,
                command=tuple(value[3]) if len(value) > 3 and value[3] else None,
            )
        if isinstance(value, str):
            return TestExecution(returncode=0, output=value)
        if isinstance(value, int):
            return TestExecution(returncode=value, output="")
        if isinstance(value, dict):
            return TestExecution(
                returncode=value.get("returncode"),
                output=str(value.get("output", "")),
                timed_out=bool(value.get("timed_out", False)),
                command=tuple(value["command"]) if value.get("command") else None,
                error=value.get("error"),
            )
        return TestExecution(
            returncode=getattr(value, "returncode", None),
            output=str(getattr(value, "output", "")),
            timed_out=bool(getattr(value, "timed_out", False)),
            command=getattr(value, "command", None),
            error=getattr(value, "error", None),
        )

    def _fallback_summary(self, execution: TestExecution) -> str:
        if execution.timed_out:
            return f"Test command timed out after {self.test_timeout:g} seconds."
        if execution.returncode is None:
            return execution.error or "No test runner found."
        if execution.returncode == 0:
            return "Tests completed successfully."
        return f"Tests exited with code {execution.returncode}."

    def _fallback_failures(self, execution: TestExecution) -> list[str]:
        if execution.timed_out:
            return [f"Test command timed out after {self.test_timeout:g} seconds."]
        if execution.returncode is None:
            return [execution.error or "No test runner found."]
        if execution.returncode == 0:
            return []
        failure = f"Test process exited with code {execution.returncode}."
        if execution.output.strip():
            failure = f"{failure} Output: {execution.output.strip()[:2000]}"
        return [failure]
