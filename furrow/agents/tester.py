"""Tester agent: run the project's test suite and turn its output into a verdict.

Two design rules shape this module.

**Discovery is separate from classification.** :meth:`TesterAgent.discover_command` answers
one question -- "is there a runnable suite, and what runs it?" -- and returns ``None`` when
there is not. That ``None`` is carried through to :meth:`TesterAgent.run` as
``TestResult(skipped=True)`` and the model is never consulted, because the old code returned
the *string* ``"No test runner found."`` as if it were test output, fed it to the tester LLM,
and typically got ``passed: true`` back: a project with no tests reported success. ``skipped``
is neutral and must never be read as evidence of success.

**The process verdict is real.** Tests run with ``cwd`` set to the workspace (the old code
ran them against whatever directory the server happened to be started in), the exit code is
checked, output is decoded leniently and labelled, and a run that exceeds
``settings.test_timeout`` is killed, reaped and reported as 124 rather than silently
continuing to the next candidate command.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import get_settings
from furrow.jsonutil import parse_model_json, truncate
from furrow.llm import LLMClient
from furrow.models import TaskModel, TestResult

if TYPE_CHECKING:
    from furrow.config import Settings

__all__ = ["MAX_OUTPUT_CHARS", "TesterAgent"]

#: Combined stdout+stderr budget handed to the model. Failure detail lives at the end of a
#: run, so both ends of the output are kept (see :func:`_clip`).
MAX_OUTPUT_CHARS = 20_000

#: How many trailing output lines become `failures` when the model verdict is unusable.
MAX_FALLBACK_FAILURES = 20

#: Exit code reported for a run that exceeded ``settings.test_timeout``. 124 mirrors the
#: coreutils convention for "command timed out", which is what an operator reading the
#: summary expects, and is distinct from any real test-suite exit code.
TIMEOUT_RETURNCODE = 124

#: Exit code reported when the command could not be started at all (127 is the shell's
#: "command not found").
START_FAILURE_RETURNCODE = 127

#: Second, after :meth:`TesterAgent._execute` has killed a timed-out run, to drain the pipes.
KILL_GRACE_SECONDS = 5.0


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Shorten `text` to `limit` chars, keeping both the head and the tail.

    :func:`furrow.jsonutil.truncate` keeps only the head, which is the wrong end for test
    output: pytest prints the summary and the tracebacks last, so head-only truncation throws
    away exactly the evidence the model needs. This keeps two thirds of the budget for the
    head (import errors, collection failures) and one third for the tail (the summary and
    tracebacks), separated by an explicit omission marker.
    """
    if len(text) <= limit:
        return text
    # Leave slack for the marker so the final truncate() is a no-op guard rather than a
    # second, tail-destroying truncation.
    budget = max(limit - 80, 200)
    marker = (
        f"\n\n[... furrow omitted {len(text) - budget} chars of test output ...]\n\n"
    )
    head = budget * 2 // 3
    tail = budget - head
    return truncate(f"{text[:head]}{marker}{text[-tail:]}", limit)


def _tail_lines(text: str, count: int = MAX_FALLBACK_FAILURES) -> list[str]:
    """The last `count` non-empty lines of `text`, in order, stripped of trailing space."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-count:]


class TesterAgent:
    """Run the project's tests and classify the outcome."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        """Build a tester. `settings=None` falls back to the process-wide singleton."""
        self.settings: Settings = settings if settings is not None else get_settings()
        self.client = client or LLMClient(self.settings)
        self.workspace: Path = self.settings.workspace_path()

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run the suite for this cycle and classify the result.

        When no runnable suite is detected the model is *not* called: the result is
        ``passed=True, skipped=True``, and ``skipped`` is neutral -- it means "nothing was
        run", not "everything worked", and callers must not treat it as a regression or as
        evidence of success.

        Args:
            goal: The cycle goal, so the model can judge intent as well as output.
            tasks: The tasks executed this cycle; their ids and descriptions are included so
                the verdict can name what was supposed to work.

        Returns:
            A :class:`~furrow.models.TestResult`. If the model's reply is unusable, the
            verdict is derived from the process exit code instead.
        """
        command = self.discover_command()
        if command is None:
            where = self.workspace
            return TestResult(
                passed=True,
                skipped=True,
                summary=f"No runnable test suite detected in {where}; treating as skipped.",
                failures=[],
            )

        returncode, output = await self._execute(command)
        response = await self.client.complete(
            self._build_prompt(goal, tasks, command, returncode, output),
            system=TESTER_PROMPT,
            model=self.settings.models()["tester"],
        )
        try:
            result = parse_model_json(response, TestResult)
        except ValueError as exc:
            return self._fallback(returncode, output, exc)
        if result.skipped:
            # A command did run, so "no suite was detected" is factually wrong here; letting
            # the model set it would report a broken suite as "nothing to test".
            result.skipped = False
        return result

    def discover_command(self) -> list[str] | None:
        """Return the argv that runs this project's tests, or ``None``.

        ``None`` means *no runnable suite*, which is **not** a pass -- see
        :meth:`TesterAgent.run`, which turns it into ``skipped=True``.

        ``settings.test_command`` wins when set. Otherwise the layout is inferred, and a
        command is returned only when the project marker exists *and* the binary is on
        ``PATH`` (:func:`shutil.which`). Both halves matter: a stale ``Cargo.toml`` or a
        ``package.json`` without a test script must not be presented as this project's test
        output, and a marker whose runner is not installed must not be run at all.

        Pure and synchronous apart from reading the marker files, so it is trivially
        unit-testable with a ``tmp_path`` workspace.
        """
        override = self.settings.test_command
        if override:
            try:
                parts = shlex.split(override)
            except ValueError as exc:
                raise ValueError(
                    f"settings.test_command {override!r} is not parseable: {exc}"
                ) from exc
            return parts or None

        if self._has_pytest_layout() and shutil.which("pytest"):
            return ["pytest", "-q"]
        if self._has_npm_test_script() and shutil.which("npm"):
            return ["npm", "test", "--silent"]
        if (self.workspace / "Cargo.toml").is_file() and shutil.which("cargo"):
            return ["cargo", "test", "-q"]
        if (self.workspace / "go.mod").is_file() and shutil.which("go"):
            return ["go", "test", "./..."]
        return None

    def _has_pytest_layout(self) -> bool:
        """True when the workspace carries a pytest marker file or a conventional test dir."""
        if (self.workspace / "pytest.ini").is_file() or (self.workspace / "tox.ini").is_file():
            return True
        if self._text_has(self.workspace / "pyproject.toml", "[tool.pytest"):
            return True
        if self._text_has(self.workspace / "setup.cfg", "[tool:pytest"):
            return True
        tests = self.workspace / "tests"
        if not tests.is_dir():
            return False
        if (tests / "conftest.py").is_file():
            return True
        # `**/` matches zero directories as well, so this covers both tests/test_x.py and
        # tests/unit/test_x.py.
        return any(tests.glob("**/test_*.py"))

    def _has_npm_test_script(self) -> bool:
        """True when ``package.json`` exists and declares a non-empty ``scripts.test``."""
        try:
            raw = (self.workspace / "package.json").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        try:
            data = json.loads(raw)
        except ValueError:
            return False
        scripts = data.get("scripts") if isinstance(data, dict) else None
        return bool(isinstance(scripts, dict) and str(scripts.get("test", "")).strip())

    def _text_has(self, path: Path, marker: str) -> bool:
        """True when `path` exists and contains `marker`."""
        try:
            return marker in path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False

    async def _execute(self, command: list[str]) -> tuple[int, str]:
        """Run `command` in the workspace and return ``(returncode, output)``.

        The child gets a scrubbed copy of the environment with ``PYTHONIOENCODING=utf-8``,
        ``NO_COLOR=1``, ``FORCE_COLOR=0`` and ``CI=1`` so the model never sees ANSI escape
        sequences or progress bars: colour codes cost tokens and confuse the JSON verdict,
        and ``CI`` switches most runners to their non-interactive, plain-text output.

        Timeout handling kills the process *and* reaps it (``await proc.wait()``) so no
        zombie is left behind, and the bytes already read are still returned -- the old code
        killed and moved on to the next candidate command, discarding everything and
        presenting a *later* runner's output as this project's result.

        Returns:
            ``(returncode, combined output)``. ``124`` means the run timed out and ``127``
            means the command could not be started; neither raises.
        """
        env = dict(os.environ)
        env.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "NO_COLOR": "1",
                "FORCE_COLOR": "0",
                "CI": "1",
            }
        )
        if not command:
            return (START_FAILURE_RETURNCODE, "<no command to run>")
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except (OSError, ValueError) as exc:
            # FileNotFoundError is an OSError; ValueError covers a NUL byte in a path.
            # Neither is worth crashing a cycle over.
            return (START_FAILURE_RETURNCODE, f"<could not start {command[0]!r}: {exc}>")

        stdout, stderr = bytearray(), bytearray()
        collector = asyncio.ensure_future(_drain(proc, stdout, stderr))
        timed_out = False
        try:
            # shield() keeps the drains alive when wait_for cancels the wait, so the bytes
            # already buffered survive the kill below.
            await asyncio.wait_for(
                asyncio.shield(collector), timeout=self.settings.test_timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            with suppress(ProcessLookupError):
                proc.kill()
        await proc.wait()  # always reap: a killed child that is never waited on is a zombie
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(collector, timeout=KILL_GRACE_SECONDS)

        combined = (
            f"--- stdout ---\n{stdout.decode('utf-8', errors='replace')}\n"
            f"--- stderr ---\n{stderr.decode('utf-8', errors='replace')}"
        )
        output = _clip(combined)
        if timed_out:
            return (
                TIMEOUT_RETURNCODE,
                f"<timed out after {self.settings.test_timeout}s>\n{output}",
            )
        return proc.returncode if proc.returncode is not None else 1, output

    def _build_prompt(
        self, goal: str, tasks: list[TaskModel], command: list[str], returncode: int, output: str
    ) -> str:
        """Compose the user message: goal, executed tasks, command, exit code and output."""
        if tasks:
            done = "\n".join(f"  [{task.id}] {task.description}" for task in tasks)
        else:
            done = "  <no tasks were executed this cycle>"
        return (
            f"Goal:\n{goal}\n\n"
            f"Tasks executed this cycle:\n{done}\n\n"
            f"Command: {shlex.join(command)}\n"
            f"Working directory: {self.workspace}\n"
            f"Exit code: {returncode}\n\n"
            f"Output:\n{output}"
        )

    def _fallback(self, returncode: int, output: str, exc: ValueError) -> TestResult:
        """Verdict derived from the process exit code, used when the model's reply is unusable.

        The old fallback searched the response for the substring ``"passed"``, which gets it
        wrong in both directions: "1 failed, 3 passed" contains "passed" and is scored as a
        success, while "tests did not pass" scores as a failure. The exit code is the only
        signal here that cannot be talked around.
        """
        failures = _tail_lines(output) if returncode != 0 else []
        summary = (
            f"Tester model output was unusable ({truncate(str(exc), 200)}); falling back to "
            f"exit code {returncode}.\n{truncate(output, 1000)}"
        )
        return TestResult(
            passed=returncode == 0, summary=summary, failures=failures, skipped=False
        )


async def _drain(
    proc: asyncio.subprocess.Process, stdout: bytearray, stderr: bytearray
) -> None:
    """Read both pipes of `proc` to EOF into the given buffers.

    Draining into memory rather than using ``communicate()`` is what makes a timeout
    recoverable: ``wait_for`` cancels ``communicate()`` and its buffers go with it, whereas
    these tasks keep the bytes they have already read and finish naturally once the child is
    killed and the pipes close.
    """
    await asyncio.gather(
        _drain_stream(proc.stdout, stdout),
        _drain_stream(proc.stderr, stderr),
    )


async def _drain_stream(stream: asyncio.StreamReader | None, sink: bytearray) -> None:
    """Copy `stream` into `sink` until EOF; a missing pipe is not an error."""
    if stream is None:
        return
    while True:
        chunk = await stream.read(65_536)
        if not chunk:
            return
        sink.extend(chunk)
