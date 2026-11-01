from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Any

from furrow.agents.parsing import parse_test_result
from furrow.agents.prompts import TESTER_PROMPT, TESTER_SYSTEM
from furrow.config import Settings, TaskModel, TestResult
from furrow.config import settings as global_settings
from furrow.llm import LLMClient

# Fallback timeout for the test command when ``Settings.test_timeout`` is not
# available (the setting is read defensively, see ``resolve_test_timeout``).
DEFAULT_TEST_TIMEOUT = 600.0

# Keywords that mark a line as reporting a failure.
_FAILURE_KEYWORDS = ("fail", "error", "traceback", "assertion")

# Counter phrases that mention a failure keyword but report *zero* failures,
# e.g. "5 passed, 0 failed", "no failures", "0 errors", "failed: 0",
# "errors=0".  They describe a green run and must not be read as a failure.
_ZERO_COUNT_RE = re.compile(
    r"\b(?:0|no|none|zero)\s+(?:fail(?:ed|ure|ures)?|error(?:s)?)\b"
    r"|\b(?:fail(?:ed|ure|ures)?|error(?:s)?)\b\s*[:=]?\s*(?:0|none|zero)\b"
    r"(?=\s*[,;)]|\s*$)",
    re.IGNORECASE,
)

# A summary whose counter block is entirely zero, e.g. unittest's
# "FAILED (errors=0)" or "FAILED (failures=0, errors=0)".
_ALL_ZERO_SUMMARY_RE = re.compile(
    r"\bfail(?:ed|ure|ures)?\b\s*\(\s*"
    r"(?:\w+\s*[:=]\s*(?:0|none|zero)\s*[,;]?\s*)*\)",
    re.IGNORECASE,
)


def resolve_test_timeout(settings: Settings) -> float:
    """Return the timeout (in seconds) allowed for the test command.

    The test-suite timeout is a separate concern from the LLM request timeout:
    ``request_timeout`` bounds a single API call, while a test suite may
    legitimately run for many minutes.  ``Settings.test_timeout`` is read
    defensively (``getattr``) so the tester also works with settings objects
    that do not define the setting yet.
    """
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

    # Patterns for pytest-style failures
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

    # Pattern for file:line:col
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

    # Pattern for file:line (without col)
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

    # Extract exception types and messages
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
    __test__ = False  # not a pytest test class

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
        """Kill the test command and any grandchildren it spawned."""
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, AttributeError):
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
        else:  # pragma: no cover - non-POSIX fallback
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
        has_setup_py = (self.workspace / "setup.py").is_file()
        has_makefile = (self.workspace / "Makefile").is_file() or (
            self.workspace / "makefile"
        ).is_file()
        has_deno = (self.workspace / "deno.json").is_file() or (
            self.workspace / "deno.jsonc"
        ).is_file()
        has_bun = (self.workspace / "bun.lockb").is_file() or (
            self.workspace / "bun.lock"
        ).is_file()

        # Python projects with tests/ dir always use pytest, even if package.json exists
        if has_tests_dir and (
            has_pyproject or has_pytest_ini or has_setup_cfg or has_tox_ini or has_requirements
        ):
            return [sys.executable, "-m", "pytest", "-q"]

        # 1. tox - if tox.ini exists
        if has_tox_ini:
            return [sys.executable, "-m", "tox"]

        # 2. Makefile with a test target
        if has_makefile:
            makefile_path = (
                self.workspace / "Makefile"
                if (self.workspace / "Makefile").is_file()
                else self.workspace / "makefile"
            )
            try:
                with open(makefile_path) as fh:
                    for line in fh:
                        # Check for line starting with test: (possibly with leading whitespace)
                        stripped = line.lstrip()
                        if stripped.startswith("test:"):
                            return ["make", "test"]
            except Exception:
                pass

        # 6. pytest.ini / setup.cfg with pytest config
        if has_pytest_ini:
            return [sys.executable, "-m", "pytest", "-q"]
        if has_setup_cfg:
            try:
                with open(self.workspace / "setup.cfg") as fh:
                    cfg_content = fh.read()
                if "[tool:pytest]" in cfg_content:
                    return [sys.executable, "-m", "pytest", "-q"]
            except Exception:
                pass

        # 3. setup.py / unittest - if setup.py exists (and no pytest markers), 
        # return unittest if tests dir also exists
        if has_setup_py and has_tests_dir:
            return [sys.executable, "-m", "unittest", "discover", "-s", "tests"]

        # 4. deno
        if has_deno:
            return ["deno", "test", "-A"]

        # 5. bun
        if has_bun:
            return ["bun", "test"]

        # Check for Node.js test runners
        if has_package:
            # Check for jest/vitest configs
            for pattern in ["jest.config.*", "vitest.config.*"]:
                for _f in self.workspace.glob(pattern):
                    if (self.workspace / "pnpm-lock.yaml").is_file():
                        return ["pnpm", "test", "--silent"]
                    if (self.workspace / "yarn.lock").is_file():
                        return ["yarn", "test", "--silent"]
                    return ["npm", "test", "--silent"]
            # Check package.json for test script
            try:
                with open(self.workspace / "package.json") as fh:
                    pkg_data = json.load(fh)
                scripts = pkg_data.get("scripts", {})
                if "test" in scripts:
                    if (self.workspace / "pnpm-lock.yaml").is_file():
                        return ["pnpm", "test", "--silent"]
                    if (self.workspace / "yarn.lock").is_file():
                        return ["yarn", "test", "--silent"]
                    return ["npm", "test", "--silent"]
            except Exception:
                pass
            # Fallback: npm test if package.json exists (Python case handled above)
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
        """Combine the exit status of the test command with the LLM verdict.

        Rule: the process exit status is the ground truth.

        * the runner could not be started -> infrastructure error, ``passed=False``
        * the command exited non-zero    -> the tests failed, ``passed=False``,
          whatever the LLM says
        * the command exited zero        -> the LLM verdict decides; a "failed"
          verdict against a zero exit code is kept (a green exit code with a
          "the suite did not run" report must not be silently promoted to a
          pass) and the disagreement is explained in the summary.
        """
        if spawn_failed:
            # The runner never started: this is an infrastructure error, not a
            # test failure, and it must be reported as such.
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
            # The LLM was unavailable: fall back to the raw output.
            passed = returncode == 0 and not _has_failure_keywords(combined)
            failures = _extract_failures(combined)
            notes.append(combined[:500].strip())
        else:
            failures = list(llm_result.failures)
            if llm_result.passed and returncode != 0:
                # The exit status wins: a failed run is never reported as a pass.
                passed = False
                notes.append(
                    "The LLM reported a pass, but the non-zero exit status is authoritative."
                )
            elif not llm_result.passed and returncode == 0:
                # A green exit code with a "the suite did not run" report is not
                # silently reported as a pass.
                passed = False
                notes.append("The command exited with status 0, but the LLM reported a failure.")
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
        try:
            cmd = await self.detect_command()
            if cmd is None:
                return TestResult(
                    passed=True,
                    skipped=True,
                    summary="No test runner detected for this project (tests skipped).",
                    failures=[],
                )

            # The test-suite timeout is a separate concern from the LLM request
            # timeout: `request_timeout` bounds a single API call, while a test
            # suite may legitimately run for minutes.
            timeout = resolve_test_timeout(self.settings)

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
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **spawn_kwargs,
                )
            except Exception as exc:
                # The runner could not even be started.  That is an
                # infrastructure error, not a test failure, and it has to be
                # distinguishable from "the tests ran and failed".
                spawn_failed = True
                message = str(exc)
                spawn_error = (
                    f"{type(exc).__name__}: {message}" if message else type(exc).__name__
                )
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
                # A runner that never started has no output worth summarising.
                prompt = TESTER_PROMPT + "\n\nGoal: " + goal + "\n\nTest output:\n" + combined
                try:
                    response = await self.client.complete(
                        prompt, system=TESTER_SYSTEM, model=self.settings.tester_model
                    )
                    llm_result = parse_test_result(response)
                except Exception:
                    llm_result = None

            return self._finalize(
                llm_result=llm_result,
                returncode=returncode,
                combined=combined,
                spawn_failed=spawn_failed,
                spawn_error=spawn_error,
                timed_out=timed_out,
                timeout=timeout,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            return TestResult(
                passed=False,
                summary=f"Tester error: {message}",
                failures=[f"tester error: {message}"],
            )


def _extract_failures(output: str, limit: int = 10) -> list[str]:
    """Best-effort extraction of failure details from raw test output."""
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
    """Return True when *text* reports a failure (zero counters do not count)."""
    return any(_looks_like_failure(line) for line in text.splitlines())


def _looks_like_failure(line: str) -> bool:
    """Return True when *line* reports a real failure.

    Rule: a line only counts as a failure when it contains a failure keyword
    that is not part of a negated / zero-count phrase.  Summary lines such as
    "5 passed, 0 failed", "no failures", "0 errors", "failed: 0" or
    "FAILED (errors=0)" describe a green run, while "1 failed, 4 passed" and
    "FAILED tests/test_x.py::test_y" are real failures.
    """
    text = line.strip()
    if not text:
        return False
    if not any(k in text.lower() for k in _FAILURE_KEYWORDS):
        return False
    # Drop the all-zero summary blocks and the zero counter phrases: whatever
    # keyword survives is a genuine failure signal.
    remainder = _ALL_ZERO_SUMMARY_RE.sub(" ", text)
    remainder = _ZERO_COUNT_RE.sub(" ", remainder)
    return any(k in remainder.lower() for k in _FAILURE_KEYWORDS)
