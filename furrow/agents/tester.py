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

        # Python projects with tests/ dir always use pytest, even if package.json exists
        if has_tests_dir and (
            has_pyproject or has_pytest_ini or has_setup_cfg or has_tox_ini or has_requirements
        ):
            return [sys.executable, "-m", "pytest", "-q"]

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

            timeout = min(self.settings.request_timeout, 300)
            spawn_kwargs: dict[str, Any] = {}
            if os.name == "posix":
                spawn_kwargs["start_new_session"] = True
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(self.workspace),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **spawn_kwargs,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                except asyncio.TimeoutError:
                    await self._kill_process_tree(proc)
                    output = "Test command timed out."
                    returncode = 1
                    stdout = b""
                    stderr = output.encode()
                else:
                    returncode = proc.returncode if proc.returncode is not None else 0
            except Exception:
                returncode = 1
                stdout = b""
                stderr = b"Failed to run test command."

            combined = (
                stdout.decode("utf-8", errors="replace")
                + "\n"
                + stderr.decode("utf-8", errors="replace")
            )
            if len(combined) > 20000:
                combined = combined[:20000]

            prompt = TESTER_PROMPT + "\n\nGoal: " + goal + "\n\nTest output:\n" + combined

            try:
                response = await self.client.complete(
                    prompt, system=TESTER_SYSTEM, model=self.settings.tester_model,
                    role="tester",
                )
                result = parse_test_result(response)
                return result
            except Exception:
                passed = returncode == 0 and not _has_failure_keywords(combined)
                failures = []
                seen = set()
                # Use the new parse_test_output helper for better failure extraction
                parsed_failures = parse_test_output(combined)
                for pf in parsed_failures:
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
                        if len(failures) >= 10:
                            break
                if not failures:
                    for line in combined.splitlines():
                        ls = line.strip()
                        if ls and _looks_like_failure(ls) and ls not in seen:
                            seen.add(ls)
                            failures.append(ls)
                            if len(failures) >= 10:
                                break
                return TestResult(
                    passed=passed,
                    summary=combined[:500],
                    failures=failures,
                )
        except Exception:
            return TestResult(passed=False, summary="Tester error", failures=[])


def _has_failure_keywords(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in ("fail", "error", "traceback", "assertion"))


def _looks_like_failure(line: str) -> bool:
    lower = line.lower()
    return any(k in lower for k in ("fail", "error", "traceback", "assertion"))
