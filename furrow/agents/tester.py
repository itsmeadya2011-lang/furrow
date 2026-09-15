from __future__ import annotations

import asyncio
import json
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient


DEFAULT_TEST_TIMEOUT = 120.0
MAX_TEST_OUTPUT_CHARS = 20_000
MAX_CONTEXT_CHARS = 4_000


@dataclass
class _TestRunResult:
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    output: str = ""
    truncated: bool = False
    timed_out: bool = False
    startup_error: str | None = None
    timeout_seconds: float | None = None
    attempted_commands: list[list[str]] = field(default_factory=list)


def _bounded_text(value: object, limit: int = MAX_TEST_OUTPUT_CHARS) -> tuple[str, bool]:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)

    if len(text) <= limit:
        return text, False

    marker = "\n...[output truncated]\n"
    if limit <= len(marker):
        return text[:limit], True
    return text[: limit - len(marker)] + marker, True


def _decode_stream(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _is_credential_name(name: str) -> bool:
    upper = name.upper()
    furrow_setting = upper.startswith("FURROW_") and any(
        marker in upper for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    )
    if furrow_setting:
        return True

    return any(
        marker in upper
        for marker in (
            "API_KEY",
            "APIKEY",
            "API_TOKEN",
            "ACCESS_KEY",
            "PRIVATE_KEY",
            "AUTH_TOKEN",
            "AUTHORIZATION",
            "TOKEN",
            "SECRET",
            "PASSWORD",
            "CREDENTIAL",
        )
    )


def _subprocess_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not _is_credential_name(key)}


def _test_timeout(settings: Settings) -> float:
    try:
        timeout = float(getattr(settings, "test_timeout", DEFAULT_TEST_TIMEOUT))
    except (TypeError, ValueError):
        timeout = DEFAULT_TEST_TIMEOUT
    if timeout <= 0:
        return DEFAULT_TEST_TIMEOUT
    return timeout


def _workspace(settings: Settings) -> Path:
    try:
        return Path(getattr(settings, "workspace", Path.cwd()))
    except (TypeError, ValueError):
        return Path.cwd()


def _candidate_commands(workspace: Path) -> list[list[str]]:
    candidates: list[list[str]] = [
        ["pytest", "-q"],
        ["python", "-m", "pytest", "-q"],
    ]

    if (workspace / "package.json").is_file():
        candidates.append(["npm", "test", "--", "--silent"])
    if (workspace / "pnpm-lock.yaml").is_file() or (workspace / "pnpm-workspace.yaml").is_file():
        candidates.append(["pnpm", "test", "--", "--silent"])
    if (workspace / "yarn.lock").is_file() or (workspace / ".yarnrc").is_file() or (
        workspace / ".yarnrc.yml"
    ).is_file():
        candidates.append(["yarn", "test", "--silent"])
    if (workspace / "Cargo.toml").is_file():
        candidates.append(["cargo", "test", "-q"])
    if (workspace / "go.mod").is_file():
        candidates.append(["go", "test", "./..."])

    return candidates


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client if client is not None else LLMClient(settings=settings)
        client_settings = getattr(self.client, "settings", None)
        self.settings: Settings = settings or client_settings or Settings()
        self._last_test_result: _TestRunResult | None = None

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            test_result = await self._run_tests()
            self._last_test_result = test_result
        except Exception as exc:
            summary, _ = _bounded_text(f"Test runner failed: {exc}", MAX_CONTEXT_CHARS)
            return TestResult(passed=False, summary=summary, failures=[summary])

        if (
            test_result.returncode == 0
            and not test_result.timed_out
            and test_result.startup_error is None
        ):
            command = " ".join(test_result.command) or "test command"
            return TestResult(passed=True, summary=f"{command} passed with return code 0.")

        return await self._failed_test_result(test_result, goal, tasks)

    async def _run_tests(self) -> _TestRunResult:
        workspace = _workspace(self.settings)
        timeout = _test_timeout(self.settings)
        environment = _subprocess_environment()
        attempted: list[list[str]] = []

        for command in _candidate_commands(workspace):
            attempted.append(list(command))
            process = None
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=workspace,
                    env=environment,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=os.name != "nt",
                )
            except asyncio.TimeoutError:
                return _TestRunResult(
                    command=list(command),
                    returncode=None,
                    output=f"Timed out while starting {' '.join(command)}.",
                    startup_error=f"Timed out while starting {' '.join(command)}.",
                    timeout_seconds=timeout,
                    attempted_commands=attempted,
                )
            except Exception:
                # A missing executable is simply an unavailable runner. Other startup
                # failures are retried so a later supported runner can still be used.
                continue

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                await self._terminate_process(process)
                output = "Test command timed out."
                return _TestRunResult(
                    command=list(command),
                    returncode=getattr(process, "returncode", None),
                    output=output,
                    timed_out=True,
                    timeout_seconds=timeout,
                    attempted_commands=attempted,
                )
            except Exception as exc:
                await self._terminate_process(process)
                output, truncated = _bounded_text(f"Test command failed to communicate: {exc}")
                return _TestRunResult(
                    command=list(command),
                    returncode=getattr(process, "returncode", None),
                    output=output,
                    truncated=truncated,
                    startup_error=f"{type(exc).__name__}: {exc}",
                    timeout_seconds=timeout,
                    attempted_commands=attempted,
                )

            output, truncated = _bounded_text(_decode_stream(stdout) + _decode_stream(stderr))
            return _TestRunResult(
                command=list(command),
                returncode=process.returncode,
                output=output,
                truncated=truncated,
                timeout_seconds=timeout,
                attempted_commands=attempted,
            )

        return _TestRunResult(
            output="No supported test runner found.",
            startup_error="No supported test runner found.",
            timeout_seconds=timeout,
            attempted_commands=attempted,
        )

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        try:
            if os.name != "nt" and hasattr(process, "pid"):
                kill_signal = getattr(signal, "SIGKILL", getattr(signal, "SIGTERM", None))
                if kill_signal is not None and hasattr(os, "killpg"):
                    os.killpg(process.pid, kill_signal)
            if hasattr(process, "kill"):
                process.kill()
        except (OSError, ProcessLookupError, PermissionError, TypeError, ValueError):
            if hasattr(process, "kill"):
                try:
                    process.kill()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            await asyncio.wait_for(process.communicate(), timeout=1)
        except Exception:
            try:
                if hasattr(process, "wait"):
                    await asyncio.wait_for(process.wait(), timeout=1)
            except Exception:
                pass

    async def _failed_test_result(
        self,
        test_result: _TestRunResult,
        goal: str,
        tasks: list[TaskModel],
    ) -> TestResult:
        fallback = self._fallback_failure_result(test_result)
        if test_result.returncode is None and not test_result.timed_out:
            return fallback

        prompt = self._failure_prompt(test_result, goal, tasks)
        try:
            model = getattr(getattr(self.client, "settings", None), "tester_model", None)
            response = await self.client.complete(prompt, model=model)
        except Exception:
            return fallback

        return self._parse_failure_response(response, fallback)

    def _failure_prompt(
        self,
        test_result: _TestRunResult,
        goal: str,
        tasks: list[TaskModel],
    ) -> str:
        bounded_goal, _ = _bounded_text(goal, MAX_CONTEXT_CHARS)
        task_lines: list[str] = []
        for task in tasks[:20]:
            task_id = str(getattr(task, "id", ""))
            description, _ = _bounded_text(getattr(task, "description", ""), MAX_CONTEXT_CHARS)
            task_lines.append(f"- {task_id}: {description}")
        task_context = "\n".join(task_lines)
        if len(task_context) > MAX_CONTEXT_CHARS:
            task_context = task_context[:MAX_CONTEXT_CHARS] + "\n...[task context truncated]\n"

        command = " ".join(test_result.command) or "unknown"
        status = (
            f"timed out after {test_result.timeout_seconds:g} seconds"
            if test_result.timed_out
            else f"exited with return code {test_result.returncode}"
        )
        startup_error = test_result.startup_error or "none"
        return (
            f"{TESTER_PROMPT}\n\n"
            f"Goal:\n{bounded_goal}\n\n"
            f"Task context (do not select tests from this context):\n"
            f"{task_context or '- None'}\n\n"
            f"Test command: {command}\n"
            f"Test command {status}.\n"
            f"Startup error: {startup_error}\n\n"
            f"Bounded test output:\n{test_result.output}"
        )

    def _fallback_failure_result(self, test_result: _TestRunResult) -> TestResult:
        if test_result.timed_out:
            summary = f"Test command timed out after {test_result.timeout_seconds:g} seconds."
        elif test_result.startup_error:
            summary = f"Test runner error: {test_result.startup_error}"
        elif test_result.returncode is None:
            summary = "No supported test runner was found."
        else:
            summary = f"Test command exited with return code {test_result.returncode}."

        excerpt = self._output_excerpt(test_result.output)
        if excerpt:
            summary = f"{summary} {excerpt}"
        return TestResult(passed=False, summary=summary, failures=[summary])

    def _parse_failure_response(self, response: object, fallback: TestResult) -> TestResult:
        data: object = response
        if isinstance(response, str):
            text = response.strip()
            if text.startswith("```") and text.endswith("```"):
                lines = text.splitlines()
                if len(lines) >= 2:
                    text = "\n".join(lines[1:-1]).strip()
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError, ValueError):
                return fallback

        if not isinstance(data, dict):
            return fallback

        summary_value = data.get("summary")
        summary, _ = _bounded_text(summary_value if summary_value is not None else fallback.summary, MAX_CONTEXT_CHARS)
        failures_value = data.get("failures", [])
        if isinstance(failures_value, list):
            failures = [
                _bounded_text(item, MAX_CONTEXT_CHARS)[0]
                for item in failures_value
                if str(item).strip()
            ]
        elif failures_value is None:
            failures = []
        else:
            failures = [_bounded_text(failures_value, MAX_CONTEXT_CHARS)[0]]

        if not summary:
            summary = fallback.summary
        if not failures:
            failures = list(fallback.failures)
        return TestResult(passed=False, summary=summary, failures=failures[:20])

    def _output_excerpt(self, output: str, limit: int = 1_200) -> str:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return ""
        excerpt = "\n".join(lines[-10:])
        if len(excerpt) > limit:
            excerpt = excerpt[-limit:]
        return f"Output excerpt: {excerpt}"

