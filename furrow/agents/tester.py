from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from furrow.agents.prompts import TESTER_PROMPT, TESTER_SYSTEM_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)

# Truncate test output to last N chars to avoid blowing context window
TEST_OUTPUT_TAIL_LIMIT = 15000

TERMINATE_GRACE_SECONDS = 5.0
"""Seconds a killed test process group gets to honour SIGTERM before SIGKILL."""


def _process_group_id(proc: asyncio.subprocess.Process) -> int:
    """Return the process group id of *proc*.

    Test commands are spawned with ``start_new_session=True`` so the child calls
    ``setsid()``: it becomes both session and process-group leader and its pgid
    equals its pid.  ``os.getpgid`` is still consulted first so the group is found
    even if that assumption stops holding, and so a process that has already been
    reaped degrades to the pid instead of raising.
    """
    try:
        return os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return proc.pid


def _signal_process_group(pgid: int, sig: signal.Signals, **log_fields: Any) -> None:
    """Send *sig* to the whole process group *pgid*, tolerating a dead group.

    ``pgid <= 1`` is refused on purpose: ``killpg(0, ...)`` and ``killpg(1, ...)``
    would signal furrow's own group (and init) instead of the test tree.
    """
    name = getattr(sig, "name", sig)
    if pgid <= 1:
        log.warning("tester_killpg_refused", pgid=pgid, signal=name, **log_fields)
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        log.debug("tester_process_group_gone", pgid=pgid, signal=name, **log_fields)
    except OSError as e:
        log.warning("tester_killpg_failed", pgid=pgid, signal=name, error=str(e), **log_fields)


async def _reap(proc: asyncio.subprocess.Process) -> None:
    """Wait for *proc* so it cannot linger as a zombie."""
    with contextlib.suppress(ProcessLookupError):
        await proc.wait()


async def terminate_process_group(
    proc: asyncio.subprocess.Process,
    *,
    reason: str,
    grace: float = TERMINATE_GRACE_SECONDS,
    **log_fields: Any,
) -> None:
    """Terminate *proc*'s entire process group and reap it.

    ``pytest``, ``npm`` and ``cargo`` all fan out into a tree of grandchildren;
    signalling only the direct child leaves those holding CPU, file handles and
    the workspace.  This sends ``SIGTERM`` to the group, escalates to ``SIGKILL``
    if anything is still alive after *grace* seconds, and always reaps.

    Safe to call while handling :class:`asyncio.CancelledError`: the ``SIGTERM``
    is raised synchronously and the reap runs in its own task awaited through
    :func:`asyncio.shield`, so a cancellation delivered to the caller cannot
    interrupt the cleanup halfway and orphan the tree.  Cancellations that do
    arrive mid-cleanup are absorbed (and escalated to ``SIGKILL``) because the
    caller is already propagating its own cancellation.
    """
    pgid = _process_group_id(proc)
    log.info(
        "tester_terminating_process_group",
        reason=reason,
        pid=proc.pid,
        pgid=pgid,
        **log_fields,
    )
    _signal_process_group(pgid, signal.SIGTERM, **log_fields)

    reaper = asyncio.ensure_future(_reap(proc))
    try:
        await asyncio.wait_for(asyncio.shield(reaper), grace)
        return
    except asyncio.TimeoutError:
        log.warning("tester_process_group_escalating", reason=reason, pgid=pgid, grace=grace)
        _signal_process_group(pgid, signal.SIGKILL, **log_fields)
    except asyncio.CancelledError:
        if reaper.done():
            return
        log.warning("tester_terminating_process_group_cancelled", reason=reason, pgid=pgid)
        _signal_process_group(pgid, signal.SIGKILL, **log_fields)
    except Exception as e:  # noqa: BLE001 - cleanup must not mask the original failure
        log.warning("tester_terminate_error", reason=reason, pgid=pgid, error=str(e))
        return

    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.shield(reaper)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001 - convert any test runner error to TestResult
            return TestResult(passed=False, summary=f"Test runner error: {e}", failures=[str(e)])

        # Truncate output to tail budget
        truncated = False
        if len(test_output) > TEST_OUTPUT_TAIL_LIMIT:
            test_output = "...\n" + test_output[-TEST_OUTPUT_TAIL_LIMIT:]
            truncated = True

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        if truncated:
            prompt += "\n[Note: test output truncated to last 15000 chars]\n"

        response = await self.client.complete(
            prompt, system=TESTER_SYSTEM_PROMPT, model=self.client.settings.tester_model
        )

        try:
            data = self.client.extract_json(response)
            return TestResult(**data)
        except TypeError:
            # extract_json refuses a non-str reply with TypeError, and the
            # heuristic below is str-only (.lower(), [:500]), so a provider or
            # SDK stub that answers with e.g. an int would otherwise escape run()
            # as an exception instead of degrading to a verdict.
            reply_type = type(response).__name__
            log.warning("tester_malformed_reply", reply_type=reply_type)
            return TestResult(
                passed=False,
                summary=f"Malformed tester reply: expected a JSON object, got {reply_type}",
                failures=[f"reply was {reply_type}, not a JSON object"],
            )
        except ValueError:
            # Conservative heuristic: look for explicit failure markers first
            lowered = response.lower()
            failure_markers = ["fail", "error", "not passed", "failing", "assertion failed"]
            has_failure_marker = any(marker in lowered for marker in failure_markers)
            # Only infer success if there's an explicit "all passed" or similar
            # AND no failure markers
            explicit_success = "all passed" in lowered or "all tests passed" in lowered
            passed = explicit_success and not has_failure_marker

            return TestResult(
                passed=passed,
                summary=response[:500],
                failures=[] if passed else [response[:500]],
            )

    async def _run_tests(self) -> str:
        settings = self.client.settings
        workspace = Path(settings.workspace).resolve()

        if not workspace.exists():
            return f"Workspace does not exist: {workspace}"

        workspace.mkdir(parents=True, exist_ok=True)

        # If explicit test_command is set, use it
        if settings.test_command:
            log.info("tester_using_explicit_command", command=settings.test_command)
            return await self._run_command_shell(
                settings.test_command, workspace, settings.test_timeout
            )

        # Auto-detect test runner based on project files
        candidates = self._detect_candidates(workspace)

        for cmd, runner_name in candidates:
            log.info("tester_trying_runner", runner=runner_name, command=cmd)
            output = await self._run_command(cmd, workspace, settings.test_timeout, runner_name)
            if output is not None:
                log.info("tester_runner_succeeded", runner=runner_name)
                return output
            log.info("tester_runner_failed_or_not_found", runner=runner_name)

        # Only reached when every detected candidate returned None, i.e. each
        # runner binary could not be launched (a non-zero exit returns its output,
        # not None, so it never falls through here). The verifier confirmed this
        # output is intentionally fed to the model, not treated as a retry signal.
        return "No test runner found or all runners failed."

    def _detect_candidates(self, workspace: Path) -> list[tuple[list[str], str]]:
        """Return list of (command_list, runner_name) tuples, ordered by preference."""
        candidates: list[tuple[list[str], str]] = []

        # Check for pytest config files
        if (
            (workspace / "pyproject.toml").exists()
            or (workspace / "pytest.ini").exists()
            or (workspace / "tox.ini").exists()
        ):
            candidates.append((["pytest", "-q"], "pytest"))
            candidates.append((["python", "-m", "pytest", "-q"], "python -m pytest"))

        # Check for package.json (npm/pnpm/yarn)
        if (workspace / "package.json").exists():
            if (workspace / "pnpm-lock.yaml").exists():
                candidates.append((["pnpm", "test", "--", "--silent"], "pnpm"))
            elif (workspace / "yarn.lock").exists():
                candidates.append((["yarn", "test", "--silent"], "yarn"))
            else:
                candidates.append((["npm", "test", "--", "--silent"], "npm"))

        # Check for Cargo.toml
        if (workspace / "Cargo.toml").exists():
            candidates.append((["cargo", "test", "-q"], "cargo"))

        # Check for go.mod
        if (workspace / "go.mod").exists():
            candidates.append((["go", "test", "./..."], "go"))

        # Fallback: python -m pytest if no other runner matched
        if not candidates:
            candidates.append((["python", "-m", "pytest", "-q"], "python -m pytest (fallback)"))

        return candidates

    async def _run_command(
        self, cmd: list[str], cwd: Path, timeout: int, runner_name: str
    ) -> str | None:
        """Run a command list directly (no shell).

        Returns the runner's combined stdout+stderr output regardless of exit code,
        so the model can diagnose a genuine failure; ``None`` is returned only when
        the command could not be launched at all (``FileNotFoundError`` / ``OSError``).
        A timeout returns a descriptive timeout string rather than ``None``, and a
        cancellation is propagated to the caller.

        The caller, :meth:`_run_tests`, treats a ``None`` result as an unavailable
        runner and falls through to the next candidate, while any other result -- a
        passing run, a failing run, or a timeout notice -- ends the search and is
        fed to the model as the verdict; a non-zero exit is therefore surfaced by
        design instead of triggering another candidate.

        The runner starts in its own process group (``start_new_session=True``) so a
        timeout or a cancellation can take down the whole runner tree; see
        :func:`terminate_process_group`.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.CancelledError:
                await terminate_process_group(proc, reason="cancelled", runner=runner_name)
                raise
            except asyncio.TimeoutError:
                await terminate_process_group(proc, reason="timeout", runner=runner_name)
                log.warning("tester_timeout", runner=runner_name, timeout=timeout)
                return f"Test runner '{runner_name}' timed out after {timeout}s."

            output = stdout.decode(errors="replace") + stderr.decode(errors="replace")

            if proc.returncode == 0:
                log.info("tester_runner_passed", runner=runner_name, exit_code=proc.returncode)
                return output
            else:
                log.info("tester_runner_failed", runner=runner_name, exit_code=proc.returncode)
                return output  # Return output even on failure so LLM can analyze it

        except FileNotFoundError:
            log.debug("tester_runner_not_found", runner=runner_name)
            return None
        except OSError as e:
            log.warning("tester_runner_error", runner=runner_name, error=str(e))
            return None

    async def _run_command_shell(self, command: str, cwd: Path, timeout: int) -> str:
        """Run a shell command string (for explicit test_command override).

        The shell starts in its own process group (``start_new_session=True``) so
        cancelling this coroutine tears down the ``sh -c`` process together with
        everything it spawned; see :func:`terminate_process_group`.
        """
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.CancelledError:
                await terminate_process_group(proc, reason="cancelled", command=command)
                raise
            except asyncio.TimeoutError:
                await terminate_process_group(proc, reason="timeout", command=command)
                log.warning("tester_shell_timeout", command=command, timeout=timeout)
                return f"Test command timed out after {timeout}s: {command}"

            output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
            log.info("tester_shell_done", command=command, exit_code=proc.returncode)
            return output

        except OSError as e:
            log.error("tester_shell_error", command=command, error=str(e))
            return f"Test command failed to execute: {e}"
