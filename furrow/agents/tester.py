from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


DEFAULT_TEST_TIMEOUT = 120.0
MAX_TEST_OUTPUT_CHARS = 20 * 1024
MAX_CONTEXT_CHARS = 4 * 1024


@dataclass
class TestExecution:
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    output: str = ""
    truncated: bool = False
    timed_out: bool = False
    startup_error: str | None = None
    error: str | None = None
    timeout_seconds: float | None = None


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


def _setting_float(settings: Settings, name: str, default: float) -> float:
    try:
        value = float(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _workspace(settings: Settings) -> Path:
    try:
        return Path(getattr(settings, "workspace", Path.cwd())).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return Path.cwd()


def _configured_command(settings: Settings) -> list[str] | None:
    value = getattr(settings, "test_command", None)
    if value is None:
        return None
    if isinstance(value, str):
        try:
            command = shlex.split(value)
        except ValueError:
            return None
    elif isinstance(value, (list, tuple)):
        command = [str(item) for item in value]
    else:
        return None
    if not command or any(not item or "\x00" in item for item in command):
        return None
    return command


def _candidate_commands(workspace: Path) -> list[list[str]]:
    candidates: list[list[str]] = []
    python_markers = (
        "pytest.ini",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
        "tests",
        "test",
    )
    if any((workspace / marker).exists() for marker in python_markers) and shutil.which("pytest"):
        candidates.append(["pytest", "-q"])
    if (workspace / "package.json").is_file():
        if shutil.which("npm"):
            candidates.append(["npm", "test", "--", "--silent"])
        if shutil.which("pnpm") and (
            (workspace / "pnpm-lock.yaml").is_file()
            or (workspace / "pnpm-workspace.yaml").is_file()
        ):
            candidates.append(["pnpm", "test", "--", "--silent"])
        if shutil.which("yarn") and (
            (workspace / "yarn.lock").is_file()
            or (workspace / ".yarnrc").is_file()
            or (workspace / ".yarnrc.yml").is_file()
        ):
            candidates.append(["yarn", "test", "--silent"])
    if (workspace / "Cargo.toml").is_file() and shutil.which("cargo"):
        candidates.append(["cargo", "test", "-q"])
    if (workspace / "go.mod").is_file() and shutil.which("go"):
        candidates.append(["go", "test", "./..."])
    if not candidates and shutil.which("pytest"):
        candidates.append(["pytest", "-q"])
    return candidates


def _extract_json_object(response: object) -> str:
    if isinstance(response, bytes):
        text = response.decode("utf-8", errors="replace")
    else:
        text = str(response)
    text = text.lstrip("﻿").strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("response does not contain a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("response contains an unbalanced JSON object")


def _parse_test_response(response: object) -> TestResult:
    data = json.loads(_extract_json_object(response))
    if not isinstance(data, dict):
        raise ValueError("test result must be a JSON object")
    if type(data.get("passed")) is not bool:
        raise ValueError("passed must be a boolean")
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("summary must be a non-empty string")
    failures = data.get("failures", [])
    if not isinstance(failures, list) or not all(type(item) is str for item in failures):
        raise ValueError("failures must be an array of strings")
    return TestResult(
        passed=data["passed"],
        summary=summary.strip(),
        failures=[failure for failure in failures if failure.strip()][:20],
    )


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        client_settings = getattr(self.client, "settings", None)
        self.settings: Settings = settings or client_settings or default_settings
        self._last_test_result: TestExecution | None = None
        self.last_execution: TestExecution | None = None

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        try:
            test_result = await self._run_tests()
            self._last_test_result = self._execution_from_output(test_result)
            self.last_execution = self._last_test_result
        except Exception as exc:
            summary, _ = _bounded_text(f"Test runner failed: {exc}", MAX_CONTEXT_CHARS)
            return TestResult(passed=False, summary=summary, failures=[summary])

        execution = self._last_test_result
        if (
            execution.returncode == 0
            and not execution.timed_out
            and execution.startup_error is None
        ):
            command = " ".join(execution.command) or "test command"
            return TestResult(passed=True, summary=f"{command} passed with exit code 0.")

        return await self._failed_test_result(execution, goal, tasks)

    def _execution_from_output(self, output: object) -> TestExecution:
        text = _decode_stream(output).strip()
        returncode: int | None = None
        timed_out = "timed out" in text.lower()
        startup_error: str | None = None
        command: list[str] = []
        for line in text.splitlines():
            if line.startswith("Command: "):
                command = shlex.split(line[len("Command: ") :])
            elif line.startswith("Exit code: "):
                try:
                    returncode = int(line[len("Exit code: ") :].strip())
                except ValueError:
                    pass
            elif line.startswith("Test command could not be started: "):
                startup_error = line[len("Test command could not be started: ") :]
        if not text:
            startup_error = "No test output was produced."
        return TestExecution(
            command=command,
            returncode=returncode,
            output=text,
            timed_out=timed_out,
            startup_error=startup_error,
            timeout_seconds=_setting_float(self.settings, "test_timeout", DEFAULT_TEST_TIMEOUT),
        )

    async def _run_tests(self) -> str:
        workspace = _workspace(self.settings)
        timeout = _setting_float(self.settings, "test_timeout", DEFAULT_TEST_TIMEOUT)
        configured = _configured_command(self.settings)
        commands = [configured] if configured is not None else _candidate_commands(workspace)

        for command in commands:
            process = None
            stdout = b""
            stderr = b""
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=os.name != "nt",
                )
            except TypeError:
                try:
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        cwd=workspace,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                except Exception:
                    continue
            except Exception:
                continue

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                await self._terminate_process(process)
                output = (
                    f"Command: {shlex.join(command)}\n"
                    f"Test command timed out after {timeout:g} seconds.\n"
                    f"Exit code: {getattr(process, 'returncode', None)}\n"
                    f"Stdout:\n{_decode_stream(stdout) if 'stdout' in locals() else ''}"
                    f"Stderr:\n{_decode_stream(stderr) if 'stderr' in locals() else ''}"
                )
                self._last_test_result = TestExecution(
                    command=list(command),
                    returncode=getattr(process, "returncode", None),
                    output=output,
                    timed_out=True,
                    timeout_seconds=timeout,
                )
                return output
            except Exception as exc:
                await self._terminate_process(process)
                output, _ = _bounded_text(f"Test command failed to communicate: {exc}")
                self._last_test_result = TestExecution(
                    command=list(command),
                    returncode=getattr(process, "returncode", None),
                    output=output,
                    startup_error=f"{type(exc).__name__}: {exc}",
                    timeout_seconds=timeout,
                )
                return self._format_execution(self._last_test_result)

            output, truncated = _bounded_text(
                f"Command: {shlex.join(command)}\n"
                f"Exit code: {process.returncode}\n"
                f"Stdout:\n{_decode_stream(stdout)}\n"
                f"Stderr:\n{_decode_stream(stderr)}"
            )
            self._last_test_result = TestExecution(
                command=list(command),
                returncode=process.returncode,
                output=output,
                truncated=truncated,
                timeout_seconds=timeout,
            )
            return output

        result = TestExecution(
            output="No supported test runner found.",
            startup_error="No supported test runner found.",
            timeout_seconds=timeout,
        )
        self._last_test_result = result
        return result.output

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        try:
            if os.name != "nt" and hasattr(process, "pid") and hasattr(os, "killpg"):
                os.killpg(process.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            elif hasattr(process, "kill"):
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
                await asyncio.wait_for(process.wait(), timeout=1)
            except Exception:
                pass

    @staticmethod
    def _format_execution(result: TestExecution) -> str:
        if result.startup_error:
            return (
                f"Command: {shlex.join(result.command)}\n"
                f"Test command could not be started: {result.startup_error}\n"
                f"Exit code: {result.returncode if result.returncode is not None else 'unknown'}"
            )
        return result.output

    async def _failed_test_result(
        self,
        execution: TestExecution,
        goal: str,
        tasks: list[TaskModel],
    ) -> TestResult:
        fallback = self._fallback_failure_result(execution)
        if execution.returncode is None and not execution.timed_out:
            return fallback

        bounded_goal, _ = _bounded_text(goal, MAX_CONTEXT_CHARS)
        task_lines: list[str] = []
        for task in tasks[:20]:
            description, _ = _bounded_text(getattr(task, "description", ""), MAX_CONTEXT_CHARS)
            task_lines.append(f"- {getattr(task, 'id', '')}: {description}")
        task_context = "\n".join(task_lines)
        if len(task_context) > MAX_CONTEXT_CHARS:
            task_context = task_context[:MAX_CONTEXT_CHARS] + "\n...[task context truncated]\n"

        status = (
            f"timed out after {execution.timeout_seconds:g} seconds"
            if execution.timed_out
            else f"exited with exit code {execution.returncode}"
        )
        prompt = (
            f"{TESTER_PROMPT}\n\n"
            f"Goal:\n{bounded_goal}\n\n"
            f"Task context:\n{task_context or '- None'}\n\n"
            f"Test command: {shlex.join(execution.command) if execution.command else 'unknown'}\n"
            f"Test command {status}.\n\n"
            f"Bounded test output:\n{execution.output}"
        )
        try:
            model = getattr(self.settings, "tester_model", None) or getattr(
                getattr(self.client, "settings", None), "tester_model", None
            ) or default_settings.tester_model
            response = await self.client.complete(prompt, model=model)
        except Exception:
            return fallback

        try:
            parsed = _parse_test_response(response)
        except (json.JSONDecodeError, TypeError, ValueError):
            return fallback
        return TestResult(passed=False, summary=parsed.summary, failures=parsed.failures)

    def _fallback_failure_result(self, execution: TestExecution) -> TestResult:
        if execution.timed_out:
            summary = f"Test command timed out after {execution.timeout_seconds:g} seconds."
        elif execution.startup_error:
            summary = f"Test runner error: {execution.startup_error}"
        elif execution.returncode is None:
            summary = "No supported test runner was found."
        else:
            summary = f"Test command exited with exit code {execution.returncode}."
        excerpt = self._output_excerpt(execution.output)
        if excerpt:
            summary = f"{summary} {excerpt}"
        return TestResult(passed=False, summary=summary, failures=[summary])

    @staticmethod
    def _output_excerpt(output: str, limit: int = 1_200) -> str:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return ""
        excerpt = "\n".join(lines[-10:])
        if len(excerpt) > limit:
            excerpt = excerpt[-limit:]
        return f"Output excerpt: {excerpt}"
