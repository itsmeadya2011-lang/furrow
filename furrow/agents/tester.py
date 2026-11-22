from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

from furrow.agents.parsing import parse_test_result
from furrow.agents.prompts import TESTER_PROMPT, TESTER_SYSTEM
from furrow.config import Settings, TaskModel, TestResult
from furrow.config import settings as global_settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TEST_TIMEOUT = 600.0

_FAILURE_KEYWORDS = (
    "fail",
    "error",
    "traceback",
    "assertion",
    "panic",
    "segfault",
    "segmentation fault",
    "core dumped",
    "fatal error",
)

_ZERO_COUNT_RE = re.compile(
    r"\b(?:0|no|none|zero)\s+(?:fail(?:ed|ure|ures)?|error(?:s)?)\b"
    r"|\b(?:fail(?:ed|ure|ures)?|error(?:s)?)\b\s*[:=]?\s*(?:0|none|zero)\b"
    r"(?=\s*[,;)]|\s*$)",
    re.IGNORECASE,
)

_ALL_ZERO_SUMMARY_RE = re.compile(
    r"\bfail(?:ed|ure|ures)?\b\s*\(\s*"
    r"(?:\w+\s*[:=]\s*(?:0|none|zero)\s*[,;]?\s*)*\)",
    re.IGNORECASE,
)


def resolve_test_timeout(settings: Settings) -> float:
    raw = getattr(settings, "test_timeout", None)
    try:
        timeout = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        timeout = DEFAULT_TEST_TIMEOUT
    if timeout <= 0:
        timeout = DEFAULT_TEST_TIMEOUT
    return timeout


def parse_test_output(output: str) -> list[dict]:
    """Parse raw test output and return structured failure information."""
    failures: list[dict] = []
    seen_locations = set()

    pytest_failed_pattern = re.compile(
        r"^FAILED\s+([^\s]+)::([^\s]+)\s*-?\s*(.*)$", re.MULTILINE
    )
    for match in pytest_failed_pattern.finditer(output):
        test_path = match.group(1)
        test_name = match.group(2)
        error_msg = match.group(3).strip() if match.group(3) else ""

        file_path = test_path
        if "::" in file_path:
            file_path = file_path.split("::")[0]

        location_key = f"{file_path}::{test_name}"
        if location_key not in seen_locations:
            failures.append(
                {
                    "file": file_path,
                    "line": None,
                    "test": test_name,
                    "error": error_msg,
                }
            )
            seen_locations.add(location_key)

    file_line_col_pattern = re.compile(r"([^\s:]+\.py):(\d+):(\d+)")
    for match in file_line_col_pattern.finditer(output):
        file_path = match.group(1)
        line_num = int(match.group(2))
        location_key = f"{file_path}:{line_num}"
        if location_key not in seen_locations:
            failures.append(
                {
                    "file": file_path,
                    "line": line_num,
                    "test": None,
                    "error": "",
                }
            )
            seen_locations.add(location_key)

    file_line_pattern = re.compile(r"([^\s:]+\.py):(\d+)(?!\d*:)")
    for match in file_line_pattern.finditer(output):
        file_path = match.group(1)
        line_num = int(match.group(2))
        location_key = f"{file_path}:{line_num}"
        if location_key not in seen_locations and not any(
            f.get("line") == line_num and f.get("file") == file_path for f in failures
        ):
            failures.append(
                {
                    "file": file_path,
                    "line": line_num,
                    "test": None,
                    "error": "",
                }
            )
            seen_locations.add(location_key)

    exception_pattern = re.compile(r"([A-Za-z]+Error|[A-Za-z]+Exception):\s*(.+)$", re.MULTILINE)
    if not failures:
        for match in exception_pattern.finditer(output):
            exc_type = match.group(1)
            exc_msg = match.group(2).strip()
            location_key = f"exception:{exc_type}:{exc_msg[:50]}"
            if location_key not in seen_locations:
                failures.append(
                    {
                        "file": None,
                        "line": None,
                        "test": None,
                        "error": f"{exc_type}: {exc_msg}",
                    }
                )
                seen_locations.add(location_key)
    else:
        for exc_match in exception_pattern.finditer(output):
            exc_type = exc_match.group(1)
            exc_msg = exc_match.group(2).strip()
            exc_full = f"{exc_type}: {exc_msg}"
            for failure in failures:
                if failure.get("error") == "":
                    failure["error"] = exc_full
                    break

    return failures


class TesterAgent:
    __test__ = False

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)
        self.workspace = Path(workspace) if workspace is not None else Path(self.settings.workspace)

    @staticmethod
    async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, AttributeError):
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
        else:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()

    async def detect_command(self) -> list[str] | None:
        has_package = (self.workspace / "package.json").is_file()
        has_pyproject = (self.workspace / "pyproject.toml").is_file()
        has_tests_dir = (self.workspace / "tests").is_dir()
        has_pytest_ini = (self.workspace / "pytest.ini").is_file()
        has_setup_cfg = (self.workspace / "setup.cfg").is_file()
        has_tox_ini = (self.workspace / "tox.ini").is_file()
        has_requirements = (self.workspace / "requirements.txt").is_file()

        if has_tests_dir and (
            has_pyproject or has_pytest_ini or has_setup_cfg or has_tox_ini or has_requirements
        ):
            return [sys.executable, "-m", "pytest", "-q"]

        if has_package:
            for pattern in ["jest.config.*", "vitest.config.*"]:
                for _f in self.workspace.glob(pattern):
                    if (self.workspace / "pnpm-lock.yaml").is_file():
                        return ["pnpm", "test", "--silent"]
                    if (self.workspace / "yarn.lock").is_file():
                        return ["yarn", "test", "--silent"]
                    return ["npm", "test", "--silent"]
            try:
                with open(self.workspace / "package.json", encoding="utf-8") as fh:
                    pkg_data = json.load(fh)
                scripts = pkg_data.get("scripts", {})
                if "test" in scripts:
                    if (self.workspace / "pnpm-lock.yaml").is_file():
                        return ["pnpm", "test", "--silent"]
                    if (self.workspace / "yarn.lock").is_file():
                        return ["yarn", "test", "--silent"]
                    return ["npm", "test", "--silent"]
            except (OSError, UnicodeError, ValueError) as exc:
                logger.warning("Could not read package.json: %s", exc)
            if (self.workspace / "pnpm-lock.yaml").is_file():
                return ["pnpm", "test", "--silent"]
            if (self.workspace / "yarn.lock").is_file():
                return ["yarn", "test", "--silent"]
            return ["npm", "test", "--silent"]

        python_markers = [
            "pyproject.toml",
            "pytest.ini",
            "setup.cfg",
            "tox.ini",
            "requirements.txt",
        ]
        if any((self.workspace / m).is_file() for m in python_markers):
            return [sys.executable, "-m", "pytest", "-q"]

        if (self.workspace / "Cargo.toml").is_file():
            return ["cargo", "test", "-q"]
        if (self.workspace / "go.mod").is_file():
            return ["go", "test", "./..."]

        return None

    def _finalize(
        self,
        *,
        llm_result: TestResult | None,
        returncode: int,
        combined: str,
        spawn_failed: bool,
        spawn_error: str,
        timed_out: bool,
        timeout: float,
    ) -> TestResult:
        if spawn_failed:
            summary = (
                f"The test runner could not be started: {spawn_error}. The tests did not run."
            )
            return TestResult(
                passed=False,
                summary=summary,
                failures=[f"test runner could not be started: {spawn_error}"],
            )

        if timed_out:
            status = (
                f"The test command timed out after {timeout:g}s and was killed; "
                "the test run did not finish."
            )
        elif returncode != 0:
            status = f"The test command exited with status {returncode}; the test run failed."
        else:
            status = "The test command exited with status 0."

        notes: list[str] = [status]
        if llm_result is None:
            passed = returncode == 0 and not _has_failure_keywords(combined)
            failures = _extract_failures(combined)
            notes.append(combined[:500].strip())
        else:
            failures = list(llm_result.failures)
            if llm_result.passed and returncode != 0:
                passed = False
                notes.append(
                    "The LLM reported a pass, but the non-zero exit status is authoritative."
                )
            elif not llm_result.passed and returncode == 0:
                passed = False
                notes.append("The command exited with status 0, but the LLM reported a failure.")
            elif llm_result.passed and returncode == 0 and llm_result.failures:
                passed = False
                notes.append(
                    "The LLM reported a pass but listed failing tests; "
                    "the run is treated as failed."
                )
            else:
                passed = llm_result.passed
            if not failures:
                failures = _extract_failures(combined)
            if llm_result.summary:
                notes.append(llm_result.summary)

        if not failures and not passed:
            failures = [status]

        summary = " ".join(part for part in notes if part).strip()
        return TestResult(passed=passed, summary=summary, failures=failures)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        start_time = time.monotonic()
        try:
            cmd = await self.detect_command()
            if cmd is None:
                logger.info("no_test_runner_detected", workspace=str(self.workspace))
                return TestResult(
                    passed=True,
                    skipped=True,
                    summary="No test runner detected for this project (tests skipped).",
                    failures=[],
                )

            timeout = resolve_test_timeout(self.settings)
            logger.info("test_run_started", command=" ".join(cmd), timeout=timeout)

            spawn_kwargs: dict[str, Any] = {}
            if os.name == "posix":
                spawn_kwargs["start_new_session"] = True

            spawn_failed = False
            spawn_error = ""
            timed_out = False
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(self.workspace),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **spawn_kwargs,
                )
            except Exception as exc:
                spawn_failed = True
                message = str(exc)
                spawn_error = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
                returncode = 1
                stdout = b""
                stderr = b""
            else:
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                except asyncio.TimeoutError:
                    await self._kill_process_tree(proc)
                    timed_out = True
                    returncode = 1
                    stdout = b""
                    stderr = f"Test command timed out after {timeout:g}s.".encode()
                else:
                    returncode = proc.returncode if proc.returncode is not None else 0

            combined = (
                stdout.decode("utf-8", errors="replace")
                + "\n"
                + stderr.decode("utf-8", errors="replace")
            )
            if len(combined) > 20000:
                combined = "[... output truncated ...]\n" + combined[-20000:]

            llm_result: TestResult | None = None
            if not spawn_failed:
                prompt = TESTER_PROMPT + "\n\nGoal: " + goal + "\n\nTest output:\n" + combined
                try:
                    response = await self.client.complete(
                        prompt, system=TESTER_SYSTEM, model=self.settings.tester_model
                    )
                    llm_result = parse_test_result(response)
                except Exception:
                    logger.exception("tester_llm_parse_failed")
                    llm_result = None

            result = self._finalize(
                llm_result=llm_result,
                returncode=returncode,
                combined=combined,
                spawn_failed=spawn_failed,
                spawn_error=spawn_error,
                timed_out=timed_out,
                timeout=timeout,
            )

            duration_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "test_run_finished",
                passed=result.passed,
                skipped=result.skipped,
                duration_ms=duration_ms,
                failures_count=len(result.failures),
            )
            return result
        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.exception("tester_error", duration_ms=duration_ms)
            message = f"{type(exc).__name__}: {exc}"
            return TestResult(
                passed=False,
                summary=f"Tester error: {message}",
                failures=[f"tester error: {message}"],
            )


def _extract_failures(output: str, limit: int = 10) -> list[str]:
    failures: list[str] = []
    seen: set[str] = set()

    for pf in parse_test_output(output):
        if pf.get("test") and pf.get("error"):
            fail_str = f"{pf['test']} - {pf['error']}"
        elif pf.get("test"):
            fail_str = str(pf["test"])
        elif pf.get("error"):
            fail_str = str(pf["error"])
        elif pf.get("file") and pf.get("line") is not None:
            fail_str = f"{pf['file']}:{pf['line']}"
        elif pf.get("file"):
            fail_str = str(pf["file"])
        else:
            continue
        if fail_str and fail_str not in seen:
            seen.add(fail_str)
            failures.append(fail_str)
            if len(failures) >= limit:
                break

    if not failures:
        for line in output.splitlines():
            ls = line.strip()
            if ls and _looks_like_failure(ls) and ls not in seen:
                seen.add(ls)
                failures.append(ls)
                if len(failures) >= limit:
                    break
    return failures


def _has_failure_keywords(text: str) -> bool:
    return any(_looks_like_failure(line) for line in text.splitlines())


def _looks_like_failure(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if not any(k in text.lower() for k in _FAILURE_KEYWORDS):
        return False
    remainder = _ALL_ZERO_SUMMARY_RE.sub(" ", text)
    remainder = _ZERO_COUNT_RE.sub(" ", remainder)
    return any(k in remainder.lower() for k in _FAILURE_KEYWORDS)