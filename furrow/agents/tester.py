from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass

from furrow.config import (
    Settings,
    TaskModel,
    TestResult,
    settings as default_settings,
)
from furrow.llm import LLMClient


TEST_TIMEOUT_SECONDS = 120.0
MAX_OUTPUT_BYTES = 64 * 1024
OUTPUT_SNIPPET_BYTES = 2_000


@dataclass(frozen=True)
class _CommandResult:
    command: tuple[str, ...]
    returncode: int | None
    output: str = ""
    timed_out: bool = False
    missing: bool = False
    truncated: bool = False
    error: str = ""


class TesterAgent:
    def __init__(
        self, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.client = client
        client_settings = getattr(client, "settings", None)
        self.settings = settings if settings is not None else client_settings or default_settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        del goal, tasks

        commands = (
            (("pytest", "-q"), False),
            (("ruff", "check", "."), True),
            (("mypy", "."), True),
        )
        results: list[_CommandResult] = []
        skipped: list[str] = []

        for command, optional in commands:
            result = await self._run_command(command)
            if result.missing and optional:
                skipped.append(shlex.join(command))
                continue
            results.append(result)

        failures: list[str] = []
        for result in results:
            failure = self._failure(result)
            if failure is not None:
                failures.append(failure)
        summary = self._summary(results, skipped)
        return TestResult(passed=not failures, summary=summary, failures=failures)

    async def _run_command(self, command: tuple[str, ...]) -> _CommandResult:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.settings.workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return _CommandResult(command=command, returncode=None, missing=True)
        except OSError as exc:
            return _CommandResult(command=command, returncode=None, error=str(exc))
        except Exception as exc:
            return _CommandResult(command=command, returncode=None, error=str(exc))

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=TEST_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
            except (ProcessLookupError, OSError):
                pass
            try:
                await process.wait()
            except Exception:
                pass
            return _CommandResult(command=command, returncode=None, timed_out=True)
        except OSError as exc:
            return _CommandResult(command=command, returncode=None, error=str(exc))
        except Exception as exc:
            return _CommandResult(command=command, returncode=None, error=str(exc))

        output, truncated = self._cap_output(stdout, stderr)
        return _CommandResult(
            command=command,
            returncode=process.returncode,
            output=output,
            truncated=truncated,
        )

    @staticmethod
    def _cap_output(stdout: bytes, stderr: bytes) -> tuple[str, bool]:
        marker = "\n... [output truncated]"
        marker_bytes = len(marker.encode("utf-8"))
        if MAX_OUTPUT_BYTES <= marker_bytes:
            return marker.encode("utf-8")[:MAX_OUTPUT_BYTES].decode(
                "utf-8", errors="replace"
            ), True

        remaining = MAX_OUTPUT_BYTES - marker_bytes
        output = stdout[:remaining]
        remaining -= len(output)
        output += stderr[:remaining]
        truncated = len(stdout) + len(stderr) > MAX_OUTPUT_BYTES
        text = output.decode("utf-8", errors="replace")
        if truncated:
            text += marker
        return text, truncated

    @staticmethod
    def _snippet(output: str) -> str:
        stripped = output.strip()
        if not stripped:
            return ""

        marker = "\n... [output truncated]"
        marker_bytes = len(marker.encode("utf-8"))
        encoded = stripped.encode("utf-8")
        if len(encoded) <= OUTPUT_SNIPPET_BYTES:
            return stripped
        if OUTPUT_SNIPPET_BYTES <= marker_bytes:
            return marker.encode("utf-8")[:OUTPUT_SNIPPET_BYTES].decode(
                "utf-8", errors="replace"
            )

        available = OUTPUT_SNIPPET_BYTES - marker_bytes
        capped = encoded[:available].decode("utf-8", errors="replace")
        return f"{capped.rstrip()}{marker}"

    @classmethod
    def _failure(cls, result: _CommandResult) -> str | None:
        command = shlex.join(result.command)
        if result.timed_out:
            reason = "timed out; return code unavailable"
        elif result.missing:
            reason = "executable was not found; return code unavailable"
        elif result.error:
            reason = f"could not start; return code unavailable: {result.error}"
        elif result.returncode is None:
            reason = "did not produce a return code"
        elif result.returncode != 0:
            reason = f"failed with return code {result.returncode}"
        else:
            return None

        failure = f"{command} {reason}"
        snippet = cls._snippet(result.output)
        if snippet:
            failure += f"\n{snippet}"
        return failure

    @staticmethod
    def _summary(results: list[_CommandResult], skipped: list[str]) -> str:
        statuses: list[str] = []
        for result in results:
            command = shlex.join(result.command)
            if result.timed_out:
                status = f"{command} timed out"
            elif result.missing:
                status = f"{command} was not found"
            elif result.error:
                status = f"{command} could not start"
            elif result.returncode == 0:
                status = f"{command} passed with return code 0"
            else:
                status = f"{command} failed with return code {result.returncode}"
            statuses.append(status)

        failed = any(
            result.timed_out
            or result.missing
            or result.error
            or result.returncode != 0
            for result in results
        )
        outcome = "Validation failed" if failed else "Validation passed"
        summary = f"{outcome}: {', '.join(statuses)}."
        if skipped:
            summary += f" Skipped unavailable optional tools: {', '.join(skipped)}."
        if any(result.truncated for result in results):
            summary += " Output was truncated."
        return summary
