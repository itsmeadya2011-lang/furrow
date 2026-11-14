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

    # Java/Maven/Gradle patterns
    java_failure_pattern = re.compile(
        r"^(\s*at\s+)?([a-zA-Z_$][\w$.]*)\.([a-zA-Z_$][\w$]*)"
        r"\(([^:]+):(\d+)\)",
        re.MULTILINE,
    )
    for match in java_failure_pattern.finditer(output):
        class_name = match.group(2)
        method_name = match.group(3)
        file_path = match.group(4)
        line_num = int(match.group(5))
        location_key = f"{file_path}:{line_num}"
        if location_key not in seen_locations:
            failures.append(
                {
                    "file": file_path,
                    "line": line_num,
                    "test": f"{class_name}.{method_name}",
                    "error": "",
                }
            )
            seen_locations.add(location_key)

    # Go test patterns
    go_failure_pattern = re.compile(
        r"^--- FAIL: (\S+)\s+\((\d+\.\d+)s\)",
        re.MULTILINE,
    )
    for match in go_failure_pattern.finditer(output):
        test_name = match.group(1)
        location_key = f"go:{test_name}"
        if location_key not in seen_locations:
            failures.append(
                {
                    "file": None,
                    "line": None,
                    "test": test_name,
                    "error": "",
                }
            )
            seen_locations.add(location_key)

    # Rust/Cargo test patterns
    rust_failure_pattern = re.compile(
        r"^test (\S+) \.\.\. FAILED",
        re.MULTILINE,
    )
    for match in rust_failure_pattern.finditer(output):
        test_name = match.group(1)
        location_key = f"rust:{test_name}"
        if location_key not in seen_locations:
            failures.append(
                {
                    "file": None,
                    "line": None,
                    "test": test_name,
                    "error": "",
                }
            )
            seen_locations.add(location_key)

    # JavaScript/Jest/Vitest patterns
    js_failure_pattern = re.compile(
        r"^\s*●\s+(.+?)\s*\n\s*(.+?)(?=\n\s*●|\n\s*$)",
        re.MULTILINE | re.DOTALL,
    )
    for match in js_failure_pattern.finditer(output):
        test_name = match.group(1).strip()
        error_msg = match.group(2).strip()
        location_key = f"js:{test_name}"
        if location_key not in seen_locations:
            failures.append(
                {
                    "file": None,
                    "line": None,
                    "test": test_name,
                    "error": error_msg,
                }
            )
            seen_locations.add(location_key)

    # .NET test patterns
    dotnet_failure_pattern = re.compile(
        r"^\s+Failed\s+(\S+)\s+\[(\d+)\s+ms\]",
        re.MULTILINE,
    )
    for match in dotnet_failure_pattern.finditer(output):
        test_name = match.group(1)
        location_key = f"dotnet:{test_name}"
        if location_key not in seen_locations:
            failures.append(
                {
                    "file": None,
                    "line": None,
                    "test": test_name,
                    "error": "",
                }
            )
            seen_locations.add(location_key)

    # Ruby/RSpec patterns
    rspec_failure_pattern = re.compile(
        r"^\s+\d+\)\s+(.+?)\s*\n\s+Failure/Error:\s*(.+?)(?=\n\s+\d+\)|\n\s*$)",
        re.MULTILINE | re.DOTALL,
    )
    for match in rspec_failure_pattern.finditer(output):
        test_name = match.group(1).strip()
        error_msg = match.group(2).strip()
        location_key = f"rspec:{test_name}"
        if location_key not in seen_locations:
            failures.append(
                {
                    "file": None,
                    "line": None,
                    "test": test_name,
                    "error": error_msg,
                }
            )
            seen_locations.add(location_key)

    # PHP/PHPUnit patterns
    phpunit_failure_pattern = re.compile(
        r"^\d+\)\s+(.+?)\s*\n\s+Failed asserting",
        re.MULTILINE,
    )
    for match in phpunit_failure_pattern.finditer(output):
        test_name = match.group(1).strip()
        location_key = f"phpunit:{test_name}"
        if location_key not in seen_locations:
            failures.append(
                {
                    "file": None,
                    "line": None,
                    "test": test_name,
                    "error": "",
                }
            )
            seen_locations.add(location_key)

    # Swift/XCTest patterns
    swift_failure_pattern = re.compile(
        r"^\s+(\S+)\s+failed:\s*(.+?)(?=\n\s+\S+\s+failed:|\n\s*$)",
        re.MULTILINE | re.DOTALL,
    )
    for match in swift_failure_pattern.finditer(output):
        test_name = match.group(1).strip()
        error_msg = match.group(2).strip()
        location_key = f"swift:{test_name}"
        if location_key not in seen_locations:
            failures.append(
                {
                    "file": None,
                    "line": None,
                    "test": test_name,
                    "error": error_msg,
                }
            )
            seen_locations.add(location_key)

    # Extract exception types and messages (fallback)
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


class TestRunner:
    """Represents a test runner configuration."""

    def __init__(
        self,
        name: str,
        command: list[str],
        detection_patterns: list[str] | None = None,
        config_files: list[str] | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.detection_patterns = detection_patterns or []
        self.config_files = config_files or []

    def matches(self, workspace: Path) -> bool:
        """Check if this test runner matches the workspace."""
        # Check config files
        for config_file in self.config_files:
            if (workspace / config_file).exists():
                return True
        # Check detection patterns
        for pattern in self.detection_patterns:
            for _f in workspace.glob(pattern):
                return True
        return False


# Registry of known test runners
TEST_RUNNERS: list[TestRunner] = [
    # Python
    TestRunner(
        "pytest",
        [sys.executable, "-m", "pytest", "-q"],
        config_files=["pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini"],
    ),
    TestRunner(
        "unittest",
        [sys.executable, "-m", "unittest", "discover", "-v"],
        config_files=[],
    ),
    # Node.js
    TestRunner(
        "jest",
        ["npx", "jest", "--silent"],
        config_files=["jest.config.js", "jest.config.ts", "jest.config.json"],
    ),
    TestRunner(
        "vitest",
        ["npx", "vitest", "run", "--silent"],
        config_files=["vitest.config.js", "vitest.config.ts", "vitest.config.json"],
    ),
    TestRunner(
        "npm test",
        ["npm", "test", "--silent"],
        config_files=["package.json"],
    ),
    TestRunner(
        "pnpm test",
        ["pnpm", "test", "--silent"],
        config_files=["pnpm-lock.yaml"],
    ),
    TestRunner(
        "yarn test",
        ["yarn", "test", "--silent"],
        config_files=["yarn.lock"],
    ),
    # Rust
    TestRunner(
        "cargo test",
        ["cargo", "test", "-q"],
        config_files=["Cargo.toml"],
    ),
    # Go
    TestRunner(
        "go test",
        ["go", "test", "./..."],
        config_files=["go.mod"],
    ),
    # Java
    TestRunner(
        "maven test",
        ["./mvnw", "test", "-q"],
        config_files=["pom.xml"],
    ),
    TestRunner(
        "gradle test",
        ["./gradlew", "test", "-q"],
        config_files=["build.gradle", "build.gradle.kts"],
    ),
    # .NET
    TestRunner(
        "dotnet test",
        ["dotnet", "test", "--no-build", "-v", "q"],
        config_files=["*.csproj", "*.fsproj", "*.vbproj", "*.sln"],
    ),
    # Ruby
    TestRunner(
        "rspec",
        ["bundle", "exec", "rspec"],
        config_files=[".rspec", "Gemfile"],
    ),
    TestRunner(
        "rails test",
        ["rails", "test"],
        config_files=["config/application.rb"],
    ),
    # PHP
    TestRunner(
        "phpunit",
        ["./vendor/bin/phpunit", "--no-coverage"],
        config_files=["phpunit.xml", "phpunit.xml.dist", "composer.json"],
    ),
    # Swift
    TestRunner(
        "xcodebuild test",
        ["xcodebuild", "test", "-quiet"],
        config_files=["*.xcodeproj", "*.xcworkspace", "Package.swift"],
    ),
    # Elixir
    TestRunner(
        "mix test",
        ["mix", "test"],
        config_files=["mix.exs"],
    ),
    # Kotlin
    TestRunner(
        "kotlin test",
        ["./gradlew", "test", "-q"],
        config_files=["build.gradle.kts", "settings.gradle.kts"],
    ),
]


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

    def _detect_test_runners(self) -> list[TestRunner]:
        """Detect all matching test runners in the workspace."""
        matched = []
        for runner in TEST_RUNNERS:
            if runner.matches(self.workspace):
                matched.append(runner)
        return matched

    def _select_best_runner(self, runners: list[TestRunner]) -> TestRunner | None:
        """Select the best test runner from the matched ones.

        Priority order:
        1. Explicit config files (pytest.ini, jest.config.js, etc.)
        2. Language-specific runners (Cargo.toml, go.mod, etc.)
        3. Package managers with test scripts
        """
        if not runners:
            return None

        # Priority: runners with config files first
        with_config = [r for r in runners if r.config_files and any((self.workspace / f).exists() for f in r.config_files)]
        if with_config:
            return with_config[0]

        # Priority: language-specific runners
        language_priority = {
            "cargo test": 10,
            "go test": 9,
            "maven test": 8,
            "gradle test": 7,
            "dotnet test": 6,
            "rspec": 5,
            "phpunit": 4,
            "xcodebuild test": 3,
            "mix test": 2,
            "pytest": 1,
        }
        runners.sort(key=lambda r: language_priority.get(r.name, 0), reverse=True)
        return runners[0]

    async def detect_command(self) -> list[str] | None:
        runners = self._detect_test_runners()
        best = self._select_best_runner(runners)
        if best is None:
            return None

        # Special handling for npm/pnpm/yarn based on lock files
        if best.name in ("npm test", "pnpm test", "yarn test"):
            if (self.workspace / "pnpm-lock.yaml").is_file():
                return ["pnpm", "test", "--silent"]
            if (self.workspace / "yarn.lock").is_file():
                return ["yarn", "test", "--silent"]
            return ["npm", "test", "--silent"]

        # Special handling for Maven/Gradle wrapper
        if best.name == "maven test" and not (self.workspace / "mvnw").exists():
            return ["mvn", "test", "-q"]
        if best.name == "gradle test" and not (self.workspace / "gradlew").exists():
            return ["gradle", "test", "-q"]

        # Special handling for PHPUnit
        if best.name == "phpunit" and not (self.workspace / "vendor/bin/phpunit").exists():
            return ["phpunit", "--no-coverage"]

        return best.command

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