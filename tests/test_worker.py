"""Tests for :mod:`furrow.agents.worker`: the edit wire format and its safety rails.

``WorkerAgent.run()`` never raises. It parses ``<furrow:file path="...">`` /
``<furrow:delete path="..."/>`` blocks out of the model's reply and applies them itself.
The behaviours pinned here are the ones that could corrupt a user's machine or hide a
failed task behind a cheerful summary:

* writes are confined to ``settings.workspace`` -- path traversal and absolute escapes
  are refused,
* ignored directories (``.git``, ``node_modules``, ...) and ``*.pyc`` are never written,
* a byte-identical rewrite is skipped, and a run that applied no edits at all always
  reports an error, and
* LLM failures are captured in ``WorkerAgent.error`` instead of propagating.

Repair cycles
-------------
The same file also pins ``WorkerAgent``'s ``failures`` context: a non-empty value
swaps ``WORKER_PROMPT`` for ``FIX_PROMPT`` and pastes the failure text in, while
changing *nothing* about how edits are parsed, written or refused. The wire protocol
and the write sandbox are asserted in both modes, so a prompt swap can never quietly
become a security or protocol regression.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from furrow.agents.prompts import FIX_PROMPT, WORKER_PROMPT
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from tests.conftest import FakeLLM

#: ``Callable`` shapes of the shared fixtures in ``tests/conftest.py``.
SettingsFactory = Callable[..., Settings]
FileWriter = Callable[..., Path]
BuildWorker = Callable[..., tuple[WorkerAgent, FakeLLM]]

#: Opening line shared by both prompts -- deliberately not a discriminator.
SHARED_OPENING = "You are a worker agent in an autonomous coding system called Furrow."

#: Only in ``FIX_PROMPT``: the framing that says the previous cycle failed.
REPAIR_FRAMING = "the verification step FAILED"

#: Only in ``WORKER_PROMPT``: the framing that says this is fresh implementation work.
IMPLEMENTATION_FRAMING = "Your job is to implement the assigned task by EDITING FILES"

#: Only in ``WorkerAgent._repair_section()`` (the section appended at runtime).
REPAIR_SECTION_MARKER = "This is a REPAIR cycle"

#: The appended rule, spanning the implicit-concatenation boundary of that section.
#: ``FIX_PROMPT`` words the same principle differently, so this string is specific to
#: the runtime section and cannot be satisfied by the base prompt alone.
DO_NOT_WEAKEN_RULE = "Do NOT delete a test, weaken an assertion, loosen a type check"

#: A red verdict as the tester would report it.
REPAIR_FAILURE = "tests/test_greet.py::test_greeting FAILED - assert 'hi' == 'hello'"


@pytest.fixture
def build_worker(tmp_workspace: Path, make_settings: SettingsFactory) -> BuildWorker:
    """Return ``(agent, client)`` wired to the isolated tmp workspace and a scripted LLM.

    ``failures`` is forwarded to :class:`WorkerAgent` verbatim, so a test can put the
    worker on a repair cycle. Every other keyword still reaches ``make_settings``.
    """

    def _build(
        response: str | Exception,
        files: Sequence[str] | None = None,
        *,
        failures: Sequence[str] | None = None,
        **overrides: object,
    ) -> tuple[WorkerAgent, FakeLLM]:
        settings = make_settings(tmp_workspace, **overrides)
        fake = FakeLLM([response], settings=settings)
        task = TaskModel(id="1", description="implement the greeting", files=list(files or []))
        return WorkerAgent(task, client=fake, settings=settings, failures=failures), fake

    return _build


def _file_block(path: str, content: str) -> str:
    """Render one ``<furrow:file>`` block the way the wire protocol specifies.

    Only the newline that frames the opening tag is added; the closing tag sits
    directly after ``content`` because the body *is* the complete new file.
    """
    return f'<furrow:file path="{path}">\n{content}</furrow:file>'


async def test_writes_file_from_edit_block(tmp_workspace: Path, build_worker: BuildWorker) -> None:
    """The block body is written verbatim, and the path is reported as changed."""
    agent, _fake = build_worker("Wrote it.\n" + _file_block("x.txt", "hello\nworld\n"))

    summary = await agent.run()

    assert (tmp_workspace / "x.txt").read_text(encoding="utf-8") == "hello\nworld\n"
    assert agent.changed_files == ["x.txt"]
    assert agent.error is None
    assert "Wrote it." in summary


async def test_accepts_json_fenced_edit_block(
    tmp_workspace: Path, build_worker: BuildWorker
) -> None:
    """A model that wraps the protocol in a ```json fence still has its edits applied.

    The block parser scans the raw reply, so the fence is tolerated rather than fatal;
    only the prose summary has the backticks stripped out of it.
    """
    response = "```json\n" + _file_block("x.txt", "fenced\n") + "\n```"
    agent, _fake = build_worker(response)

    await agent.run()

    assert (tmp_workspace / "x.txt").read_text(encoding="utf-8") == "fenced\n"
    assert agent.changed_files == ["x.txt"]
    assert agent.error is None


async def test_creates_missing_parent_directories(
    tmp_workspace: Path, build_worker: BuildWorker
) -> None:
    """Writing to a nested new path creates the intermediate directories."""
    agent, _fake = build_worker(_file_block("deep/nested/dir/x.txt", "nested\n"))

    await agent.run()

    target = tmp_workspace / "deep" / "nested" / "dir" / "x.txt"
    assert target.read_text(encoding="utf-8") == "nested\n"
    assert agent.changed_files == ["deep/nested/dir/x.txt"]


async def test_reports_multiple_files_and_applies_in_order(
    tmp_workspace: Path, build_worker: BuildWorker
) -> None:
    """Blocks are applied in response order, so the last block for a path wins.

    ``a.txt`` is written twice: the later content must be the one on disk, and the
    reported change list keeps first-seen order while staying duplicate-free.
    """
    response = "\n".join(
        [
            _file_block("a.txt", "first\n"),
            _file_block("b.txt", "other\n"),
            _file_block("a.txt", "second\n"),
        ]
    )
    agent, _fake = build_worker(response)

    await agent.run()

    assert (tmp_workspace / "a.txt").read_text(encoding="utf-8") == "second\n"
    assert (tmp_workspace / "b.txt").read_text(encoding="utf-8") == "other\n"
    assert agent.changed_files == ["a.txt", "b.txt"]
    assert agent.error is None


async def test_rejects_path_traversal_outside_workspace(
    tmp_workspace: Path, tmp_path: Path, build_worker: BuildWorker
) -> None:
    """SECURITY REGRESSION: ``../../`` in a ``path`` attribute must not escape the root.

    The escape target resolves to ``<tmp_path>/../etc/evil`` -- outside the workspace but
    inside the per-test pytest temp tree -- so the assertion is exact rather than "not
    somewhere in /tmp". Nothing is created anywhere and the refusal is reported.
    """
    response = _file_block("../../etc/evil", "pwned\n")
    agent, _fake = build_worker(response)
    escape_target = (tmp_workspace / "../../etc/evil").resolve()

    await agent.run()

    assert escape_target == tmp_path.parent / "etc" / "evil"
    assert not escape_target.exists()
    assert list(tmp_workspace.iterdir()) == []
    assert agent.changed_files == []
    assert agent.error is not None
    assert "refused" in agent.error


async def test_rejects_absolute_path_outside_workspace(
    tmp_workspace: Path, tmp_path: Path, build_worker: BuildWorker
) -> None:
    """An absolute ``path`` outside the workspace root is refused, not truncated."""
    outside = tmp_path / "outside" / "evil.txt"
    agent, _fake = build_worker(_file_block(str(outside), "pwned\n"))

    await agent.run()

    assert not outside.exists()
    assert not outside.parent.exists()
    assert agent.changed_files == []
    assert agent.error is not None
    assert "refused" in agent.error


@pytest.mark.parametrize(
    "path",
    [
        ".git/config",
        "node_modules/x.js",
        "__pycache__/module.pyc",
        "build/out.txt",
        ".venv/lib/pkg.py",
        "src/stale.pyc",
    ],
)
async def test_skips_ignored_directories(tmp_workspace: Path, build_worker: BuildWorker, path: str):
    """Ignored directories and ``*.pyc`` files are refused wherever they appear."""
    agent, _fake = build_worker(_file_block(path, "should not exist\n"))

    await agent.run()

    assert not (tmp_workspace / path).exists()
    assert agent.changed_files == []
    assert agent.error is not None


async def test_skips_byte_identical_write(
    tmp_workspace: Path,
    write_file: FileWriter,
    monkeypatch: pytest.MonkeyPatch,
    build_worker: BuildWorker,
) -> None:
    """A byte-identical rewrite is a no-op: nothing is written and nothing is reported.

    The spy proves the filesystem was not touched at all, rather than trusting the
    reported change list alone.
    """
    existing = write_file("x.txt", "same\n")
    before = existing.stat().st_mtime_ns
    agent, fake = build_worker(_file_block("x.txt", "same\n"))
    written: list[Path] = []

    async def recording_write(path: str | Path, content: str) -> None:
        written.append(Path(path))
        await FakeLLM.write_file(fake, path, content)

    monkeypatch.setattr(fake, "write_file", recording_write)

    await agent.run()

    assert written == []
    assert agent.changed_files == []
    assert existing.read_text(encoding="utf-8") == "same\n"
    assert existing.stat().st_mtime_ns == before
    # A run that changed nothing is a failed run, never a silent success.
    assert agent.error is not None


async def test_applies_delete_block(
    tmp_workspace: Path, write_file: FileWriter, build_worker: BuildWorker
) -> None:
    """A ``<furrow:delete>`` block unlinks the file and reports it as deleted."""
    doomed = write_file("old.txt", "obsolete\n")
    agent, _fake = build_worker('Removing it.\n<furrow:delete path="old.txt"/>')

    await agent.run()

    assert not doomed.exists()
    assert agent.deleted_files == ["old.txt"]
    assert agent.changed_files == ["old.txt"]
    assert agent.error is None


async def test_delete_then_rewrite_applies_in_order(
    tmp_workspace: Path, write_file: FileWriter, build_worker: BuildWorker
) -> None:
    """A delete followed by a write of the same path ends with the file present."""
    target = write_file("x.txt", "old\n")
    response = '<furrow:delete path="x.txt"/>' + _file_block("x.txt", "new\n")
    agent, _fake = build_worker(response)

    await agent.run()

    assert target.is_file()
    assert target.read_text(encoding="utf-8") == "new\n"
    assert agent.deleted_files == ["x.txt"]
    assert agent.changed_files == ["x.txt"]
    assert agent.error is None


async def test_normalizes_crlf(tmp_workspace: Path, build_worker: BuildWorker) -> None:
    """CRLF (and lone CR) in the block body is normalised to LF before writing."""
    response = '<furrow:file path="x.txt">\r\nalpha\r\nbeta\r\n</furrow:file>'
    agent, _fake = build_worker(response)

    await agent.run()

    target = tmp_workspace / "x.txt"
    assert target.read_bytes() == b"alpha\nbeta\n"
    assert b"\r" not in target.read_bytes()
    assert agent.changed_files == ["x.txt"]


async def test_prose_only_response_sets_error_and_changes_nothing(
    tmp_workspace: Path, build_worker: BuildWorker
) -> None:
    """A reply with no protocol block changes nothing and reports an error."""
    agent, _fake = build_worker("I looked at the code and everything already works.")

    summary = await agent.run()

    assert agent.error is not None
    assert "no edit could be applied" in agent.error
    assert agent.changed_files == []
    assert agent.deleted_files == []
    assert list(tmp_workspace.iterdir()) == []
    assert "No file changes were applied." in summary


async def test_llm_exception_is_captured_not_raised(build_worker: BuildWorker) -> None:
    """An exploding LLM client surfaces as ``error``; ``run()`` still returns normally."""
    agent, _fake = build_worker(RuntimeError("provider is down"))

    summary = await agent.run()

    assert isinstance(summary, str)
    assert agent.error is not None
    assert "the model request failed" in agent.error
    assert "provider is down" in agent.error
    assert agent.changed_files == []


async def test_prompt_contains_project_context(
    tmp_workspace: Path, write_file: FileWriter, build_worker: BuildWorker
) -> None:
    """The prompt carries the workspace root, the task's files and a missing-file note."""
    write_file("existing.txt", "seed content\n")
    agent, fake = build_worker(
        _file_block("existing.txt", "seed content\n"),
        files=["existing.txt", "missing.py"],
    )

    await agent.run()

    prompt = fake.prompts[0]
    assert f"Workspace root: {tmp_workspace}" in prompt
    assert "existing.txt" in prompt
    assert "missing.py" in prompt
    assert "seed content" in prompt
    assert "does not exist" in prompt


async def test_no_changes_when_task_files_empty(
    tmp_workspace: Path, build_worker: BuildWorker
) -> None:
    """With no declared files the worker still applies edits inside the workspace."""
    agent, fake = build_worker(_file_block("x.txt", "unconstrained\n"), files=[])

    await agent.run()

    assert (tmp_workspace / "x.txt").read_text(encoding="utf-8") == "unconstrained\n"
    assert agent.changed_files == ["x.txt"]
    assert agent.error is None
    assert "any path in the workspace" in fake.prompts[0]


async def test_uses_worker_model(tmp_workspace: Path, build_worker: BuildWorker) -> None:
    """The request is issued with the role-resolved ``worker_model``, not the default."""
    agent, fake = build_worker(_file_block("x.txt", "x\n"), worker_model="claude-worker-1")

    await agent.run()

    assert fake.models == ["claude-worker-1"]
    assert fake.models == [fake.settings.model_for("worker")]


# --------------------------------------------------------------------------- #
# Repair cycles: the FIX_PROMPT swap, and everything it must NOT change
# --------------------------------------------------------------------------- #


async def test_repair_cycle_uses_fix_prompt(build_worker: BuildWorker) -> None:
    """A repair cycle is framed by ``FIX_PROMPT`` and carries the failure text.

    Asserted against the real prompt captured by ``FakeLLM.prompts``, not against
    ``_base_prompt()`` in isolation. The first assertion is the one with teeth: the
    prompt must *begin* with ``FIX_PROMPT`` verbatim, so a ``_base_prompt()`` that
    returned ``WORKER_PROMPT`` (the pre-feature behavior) fails on that line alone --
    the two prompts share an opening line and differ from their second paragraph on.
    """
    failures = [
        REPAIR_FAILURE,
        "tests/test_greet.py::test_farewell ERROR - NameError: name 'goodby' is not defined",
    ]
    agent, fake = build_worker(_file_block("greet.py", "fixed\n"), failures=failures)

    await agent.run()

    assert agent.failures == failures
    prompt = fake.prompts[0]
    # TEETH: breaks on line 1 if _base_prompt() returns WORKER_PROMPT.
    assert prompt.startswith(FIX_PROMPT)
    assert not prompt.startswith(WORKER_PROMPT)
    # The repair framing replaced the implementation framing, not merely joined it.
    assert REPAIR_FRAMING in prompt
    assert IMPLEMENTATION_FRAMING not in prompt
    # Both prompts open identically, which is why `startswith` is asserted on the
    # whole prompt text rather than on its first line.
    assert prompt.startswith(SHARED_OPENING)
    # Every failure line is pasted in, verbatim and bulleted.
    for line in failures:
        assert f"  - {line}" in prompt
    # The do-not-weaken-a-test rule is spelled out, not just implied by FIX_PROMPT.
    assert REPAIR_SECTION_MARKER in prompt
    assert DO_NOT_WEAKEN_RULE in prompt


@pytest.mark.parametrize(
    "failures",
    [None, [], ["", "   ", "\n\t"]],
    ids=["omitted", "empty-list", "blank-only"],
)
async def test_normal_cycle_uses_worker_prompt(
    failures: Sequence[str] | None, build_worker: BuildWorker
) -> None:
    """No failures, empty failures and blank-only failures all mean a NORMAL cycle.

    A blank-only list is the sharp case: ``WorkerAgent`` drops blank entries, so
    ``self.failures`` is empty and the repair framing must not leak in.
    """
    agent, fake = build_worker(_file_block("x.txt", "written\n"), failures=failures)

    await agent.run()

    assert agent.failures == []
    prompt = fake.prompts[0]
    # TEETH: breaks if _base_prompt() returns FIX_PROMPT, or if the repair section is
    # appended on an empty failure list.
    assert prompt.startswith(WORKER_PROMPT)
    assert not prompt.startswith(FIX_PROMPT)
    assert IMPLEMENTATION_FRAMING in prompt
    assert REPAIR_FRAMING not in prompt
    assert REPAIR_SECTION_MARKER not in prompt
    assert DO_NOT_WEAKEN_RULE not in prompt


async def test_failures_default_to_empty_and_constructor_stays_backward_compatible(
    tmp_workspace: Path, make_settings: SettingsFactory
) -> None:
    """Every historical call shape still constructs, and blanks are dropped."""
    settings = make_settings(tmp_workspace)
    client = FakeLLM([], settings=settings)
    task = TaskModel(id="1", description="implement the greeting", files=[])

    # The pre-feature call shapes: positional, keyword, and no ``failures`` at all.
    shapes = [
        WorkerAgent(task, client, settings),
        WorkerAgent(task=task, client=client, settings=settings),
        WorkerAgent(task=task, client=client),
        WorkerAgent(task=task, client=client, failures=None),
        WorkerAgent(task=task, client=client, failures=[]),
    ]
    for agent in shapes:
        assert agent.failures == []
        assert agent.changed_files == []
        assert agent.deleted_files == []
        assert agent.error is None
        assert agent._base_prompt() == WORKER_PROMPT

    noisy = WorkerAgent(
        task=task,
        client=client,
        failures=[REPAIR_FAILURE, "", "   ", "\n\t", "second real failure"],
    )
    # Non-blank entries survive, in order, untrimmed; blank ones are gone.
    assert noisy.failures == [REPAIR_FAILURE, "second real failure"]
    assert noisy._base_prompt() == FIX_PROMPT

    # A list of nothing but blanks is not a repair cycle.
    only_blank = WorkerAgent(task=task, client=client, failures=["", "  ", "\n"])
    assert only_blank.failures == []
    assert only_blank._base_prompt() == WORKER_PROMPT


@pytest.mark.parametrize(
    "failures",
    [None, [REPAIR_FAILURE]],
    ids=["normal-cycle", "repair-cycle"],
)
async def test_repair_prompt_does_not_change_edit_protocol_behavior(
    tmp_workspace: Path,
    failures: Sequence[str] | None,
    build_worker: BuildWorker,
) -> None:
    """Swapping the base prompt changes the framing, never how edits are applied.

    Run in BOTH modes on purpose: the wire protocol is shared text appended to either
    prompt, so a future edit to ``FIX_PROMPT``/``WORKER_PROMPT`` that dropped or
    corrupted ``_EDIT_PROTOCOL`` would break exactly one of the two cases here.
    """
    agent, fake = build_worker(_file_block("a.txt", "x"), failures=failures)

    summary = await agent.run()

    assert (tmp_workspace / "a.txt").read_text(encoding="utf-8") == "x"
    assert agent.changed_files == ["a.txt"]
    assert agent.error is None
    assert "Changed files:" in summary
    prompt = fake.prompts[0]
    assert "EDIT PROTOCOL" in prompt
    assert "<furrow:file path=" in prompt
    assert "Workspace root:" in prompt


async def test_repair_prompt_preserves_path_traversal_guard(
    tmp_workspace: Path, tmp_path: Path, build_worker: BuildWorker
) -> None:
    """SECURITY REGRESSION: the repair prompt does not widen the write sandbox.

    A worker told to "repair these failures" must not become a worker that can write
    outside the workspace. The escape target is exact (``<tmp_path>/escape.txt``, one
    level above the root) so the assertion is not merely "not somewhere in /tmp".
    """
    agent, _fake = build_worker(
        _file_block("../escape.txt", "pwned\n"),
        failures=[REPAIR_FAILURE],
    )
    escape_target = (tmp_workspace / "../escape.txt").resolve()

    await agent.run()

    assert escape_target == tmp_path / "escape.txt"
    assert not escape_target.exists()
    assert list(tmp_workspace.iterdir()) == []
    assert agent.changed_files == []
    assert agent.error is not None
    assert "refused" in agent.error


async def test_repair_mode_still_never_raises(build_worker: BuildWorker) -> None:
    """The never-raise contract holds on a repair cycle, not only on a normal one."""
    agent, _fake = build_worker(RuntimeError("provider is down"), failures=[REPAIR_FAILURE])

    summary = await agent.run()

    assert isinstance(summary, str)
    assert agent.error is not None
    assert "the model request failed" in agent.error
    assert "provider is down" in agent.error
    assert agent.changed_files == []
    # The repair context survives the failure: it is state, not per-run scratch.
    assert agent.failures == [REPAIR_FAILURE]
