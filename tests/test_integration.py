"""Integration tests for the two paths the unit suites can only fake.

Two gaps are closed here, and both are about production code rather than
bookkeeping:

1. ``LLMClient.read_file`` / ``write_file`` / ``list_files``.  Every other test
   in this repo drives a worker with an in-memory fake, so the real aiofiles
   write - ``p.parent.mkdir(parents=True, exist_ok=True)`` followed by an
   ``aiofiles.open(..., "w")`` - had no coverage at all, even though it is the
   only code in Furrow that mutates a user's repository.  The client below
   overrides *only* ``complete()``; the filesystem helpers are the genuine
   implementations, and every assertion reads bytes back with plain
   ``pathlib``.

2. ``server._encode_event``'s byte cap.  The cap used to be computed against
   un-escaped text, so a payload full of ``"`` / ``\\`` / control characters
   could serialise to more than ``MAX_EVENT_BYTES`` (measured 8233 > 8192).  The
   existing tests only pad with ``'t'`` and ``'x'``, which never expand under
   ``json.dumps``.  The cases below are escape-heavy on purpose and assert the
   two invariants together: the frame fits the cap *and* still parses as JSON,
   because a naive byte slice silently trades one bug for the other.

Nothing here reaches the network, and nothing is written outside ``tmp_path``.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path
from typing import Any

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient
from furrow.web import server

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run every test *inside* ``tmp_path``.

    The worker emits repository-relative paths, so a relative edit is the
    realistic case - and the one that can scribble on the checkout if a test
    forgets to isolate itself.  Making ``tmp_path`` the process cwd means a
    missing fixture can never write into the repo.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


class ScriptedClient(LLMClient):
    """A real :class:`LLMClient` with only ``complete()`` replaced.

    ``read_file``, ``write_file`` and ``list_files`` are inherited untouched, so
    a worker driven by this class performs real disk I/O through aiofiles and
    real directory pruning through ``os.walk``.
    """

    def __init__(self, replies: list[str], settings: Settings) -> None:
        # NOTE: an explicit ``Settings`` is required - ``LLMClient.__init__``
        # only falls back to the global settings for an *omitted* argument.
        super().__init__(settings=settings)
        self._replies = list(replies)
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.models: list[str | None] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        self.models.append(model)
        if not self._replies:
            raise AssertionError("the worker asked for more completions than were scripted")
        return self._replies.pop(0)


def reply(*edits: tuple[str, str], summary: str = "added the module") -> str:
    """A well-formed worker envelope, exactly as the model is asked to return."""
    return json.dumps(
        {
            "summary": summary,
            "edits": [{"path": path, "content": content} for path, content in edits],
        }
    )


def settings_for(workspace: Path) -> Settings:
    return Settings(workspace=workspace)


def snapshot(root: Path) -> dict[str, bytes]:
    """Every regular file under *root* as ``relative path -> bytes``."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def wide_event(field_value: str, *, event_type: str = "task_completed", fields: int = 40) -> dict:
    """An event whose *bounded* form still overruns the cap.

    ``_bound_value`` clips every string to ``MAX_FIELD_CHARS`` (2000), so a
    single pathological field is trimmed back under 8192 bytes and never reaches
    the fitting branch.  Spreading the payload over the field budget is what
    actually exercises the post-escape budget computation.
    """
    return {"type": event_type, "cycle": 7, **{f"field_{i}": field_value for i in range(fields)}}


# --------------------------------------------------------------------------- #
# 1. the real filesystem write path
# --------------------------------------------------------------------------- #


async def test_worker_edits_land_on_disk_byte_for_byte(_workspace: Path) -> None:
    """The headline gap: a real ``aiofiles`` write, read back with ``pathlib``."""
    body = "def add(a, b):\n    return a + b\n"
    client = ScriptedClient([reply(("pkg/mod.py", body))], settings_for(_workspace))
    agent = WorkerAgent(
        task=TaskModel(id="1", description="add add()", files=["pkg/mod.py"]),
        client=client,
    )

    result = await agent.run()

    # read the bytes back off disk with the stdlib, not with the client
    assert (_workspace / "pkg" / "mod.py").read_text() == body
    assert (_workspace / "pkg" / "mod.py").read_bytes() == body.encode("utf-8")
    assert agent.written_files == ["pkg/mod.py"]
    assert result == "added the module\nWrote: pkg/mod.py"
    assert client.models == [client.settings.worker_model]


async def test_write_file_creates_missing_parent_directories(_workspace: Path) -> None:
    """``write_file`` does ``p.parent.mkdir(parents=True, exist_ok=True)``."""
    deep = "a/b/c/d.txt"
    assert not (_workspace / "a").exists()  # precondition: nothing exists yet
    client = ScriptedClient([reply((deep, "deep\n"))], settings_for(_workspace))
    agent = WorkerAgent(task=TaskModel(id="1", description="go deep"), client=client)

    await agent.run()

    created = _workspace / deep
    assert created.is_file()
    assert created.read_text() == "deep\n"
    for part in ("a", "a/b", "a/b/c"):
        assert (_workspace / part).is_dir()


async def test_write_file_creates_parents_for_an_existing_target_dir(_workspace: Path) -> None:
    """``exist_ok=True`` also makes a *second* write into the same new tree fine."""
    client = ScriptedClient(
        [reply(("x/y/one.txt", "one\n"), ("x/y/two.txt", "two\n"))], settings_for(_workspace)
    )
    agent = WorkerAgent(task=TaskModel(id="1", description="twice"), client=client)

    await agent.run()

    assert (_workspace / "x/y/one.txt").read_text() == "one\n"
    assert (_workspace / "x/y/two.txt").read_text() == "two\n"


async def test_write_file_is_also_correct_called_directly(_workspace: Path) -> None:
    """The same guarantee for a ``Path`` argument, bypassing the worker."""
    client = ScriptedClient([], settings_for(_workspace))

    await client.write_file(_workspace / "deep/nest/leaf.py", "leaf = True\n")

    assert (_workspace / "deep/nest/leaf.py").read_text() == "leaf = True\n"


async def test_write_file_truncates_rather_than_appends(_workspace: Path) -> None:
    """A second, shorter write must replace the file, not extend it.

    This is the ``"w"`` vs ``"a"`` mode check on the real aiofiles path - the
    classic silent corruption for a file-editing agent.
    """
    target = _workspace / "src/app.py"
    client = ScriptedClient([], settings_for(_workspace))
    await client.write_file(target, "# a very long original body\n" * 50)

    await client.write_file(target, "short\n")

    assert target.read_text() == "short\n"
    assert len(target.read_text()) == len("short\n")


async def test_worker_overwrites_an_existing_file_completely(_workspace: Path) -> None:
    """Same guarantee, but driven through ``WorkerAgent.run``."""
    original = _workspace / "app.py"
    original.write_text("# the old implementation\nprint('old')\n")
    before = original.read_text()
    client = ScriptedClient([reply(("app.py", "print('new')\n"))], settings_for(_workspace))
    agent = WorkerAgent(
        task=TaskModel(id="1", description="swap the body", files=["app.py"]), client=client
    )

    await agent.run()

    assert original.read_text() == "print('new')\n"
    assert before in client.prompts[0]  # the old body really was read from disk
    assert len(original.read_text()) < len(before)


async def test_second_worker_reads_back_the_first_workers_output(_workspace: Path) -> None:
    """The multi-cycle loop: cycle 2's file context comes off real disk."""
    first = "CONST = 1\n"
    client = ScriptedClient(
        [
            reply(("pkg/base.py", first), summary="cycle 1"),
            reply(("pkg/next.py", "from base import CONST\n"), summary="cycle 2"),
        ],
        settings_for(_workspace),
    )

    cycle1 = WorkerAgent(
        task=TaskModel(id="1", description="add base", files=["pkg/base.py"]), client=client
    )
    await cycle1.run()

    # cycle 2 is handed the file cycle 1 produced, and the client reads it for real
    cycle2 = WorkerAgent(
        task=TaskModel(id="2", description="use base", files=["pkg/base.py"]),
        client=client,
    )
    await cycle2.run()

    assert (_workspace / "pkg/base.py").read_text() == first
    assert (_workspace / "pkg/next.py").read_text() == "from base import CONST\n"
    assert len(client.prompts) == 2
    assert first in client.prompts[1]
    assert "pkg/base.py" in client.prompts[1]


async def test_worker_dry_run_leaves_the_tree_byte_identical(_workspace: Path) -> None:
    """``dry_run=True`` must not create, truncate or touch a single byte."""
    (_workspace / "keep.py").write_text("keep me\n")
    (_workspace / "pkg").mkdir()
    (_workspace / "pkg" / "existing.py").write_text("existing\n")
    before = snapshot(_workspace)

    client = ScriptedClient(
        [reply(("pkg/new.py", "new\n"), ("keep.py", "clobbered\n"), ("a/b/c.txt", "deep\n"))],
        settings_for(_workspace),
    )
    agent = WorkerAgent(
        task=TaskModel(id="1", description="do it all", files=["keep.py", "pkg/existing.py"]),
        client=client,
        dry_run=True,
    )

    result = await agent.run()

    assert snapshot(_workspace) == before
    assert not (_workspace / "pkg/new.py").exists()
    assert not (_workspace / "a").exists()
    assert (_workspace / "keep.py").read_text() == "keep me\n"
    assert agent.written_files == []
    # the plan is still reported, and the files were still read for context
    assert result == "added the module\nWrote: pkg/new.py, keep.py, a/b/c.txt"
    assert "keep me" in client.prompts[0]


async def test_worker_writes_relative_to_the_configured_workspace(_workspace: Path) -> None:
    """``settings.workspace`` and the worker's cwd agree on where files land.

    ``TesterAgent`` runs pytest with ``cwd=settings.workspace``; this pins the
    other half of that contract - a worker's relative edits land inside exactly
    the workspace the tester will be pointed at, and the workspace listing sees
    them.
    """
    settings = settings_for(_workspace)
    client = ScriptedClient([reply(("pkg/mod.py", "x = 1\n"))], settings)
    agent = WorkerAgent(task=TaskModel(id="1", description="add a module"), client=client)

    await agent.run()

    assert settings.workspace == _workspace
    assert (_workspace / "pkg/mod.py").read_text() == "x = 1\n"
    assert client.list_files(settings.workspace) == ["pkg/mod.py"]


async def test_worker_reads_real_file_context_for_unicode_sources(_workspace: Path) -> None:
    """Non-ASCII survives the real aiofiles read/write round trip."""
    body = '# -*- coding: utf-8 -*-\nGREETING = "\u00e9\u4e2d\U0001f600"\n'
    source = _workspace / "src/i18n.py"
    source.parent.mkdir(parents=True)
    source.write_text(body, encoding="utf-8")

    client = ScriptedClient(
        [reply(("src/i18n.py", body + "EXTRA = 1\n"))], settings_for(_workspace)
    )
    agent = WorkerAgent(
        task=TaskModel(id="1", description="extend i18n", files=["src/i18n.py"]), client=client
    )

    await agent.run()

    assert source.read_text(encoding="utf-8") == body + "EXTRA = 1\n"
    assert source.read_bytes() == (body + "EXTRA = 1\n").encode("utf-8")
    assert body in client.prompts[0]  # read back through the real reader


def test_list_files_prunes_ignored_trees_on_a_real_tree(_workspace: Path) -> None:
    """``os.walk`` + in-place ``dirnames[:]`` pruning, against a real directory.

    The unit test covers the same rule with a fake; this one proves the pruning
    survives a genuine nested tree, including an ignored directory that is *not*
    at the root and a file whose *name* merely looks like an ignored entry.
    """
    root = _workspace
    real_files = [
        "main.py",
        "README.md",
        "src/app.py",
        "src/deep/nested/util.py",
        "dist_sibling/keep.py",  # only *starts* with an ignored name
        "src/notes.pyc.bak",  # only *ends* with an ignored suffix's neighbour
    ]
    ignored_files = [
        ".git/config",
        "__pycache__/x.pyc",
        "node_modules/a.js",
        ".venv/lib.py",
        "dist/out.js",
        "build/o.js",
        "src/__pycache__/nested.pyc",  # ignored dir below the root, not at it
        "src/node_modules/pkg/index.js",
    ]
    for name in real_files + ignored_files:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x\n")

    client = ScriptedClient([], settings_for(root))
    files = client.list_files(root)

    assert files == sorted(real_files)
    assert files == sorted(files)  # deterministic ordering
    for name in ignored_files:
        assert name not in files
        assert not any(Path(name).name in Path(found).parts for found in files)


async def test_list_files_reflects_what_the_worker_actually_wrote(_workspace: Path) -> None:
    """The listing and the real writes agree - no drift, no stale entries."""
    settings = settings_for(_workspace)
    client = ScriptedClient(
        [reply(("src/one.py", "1\n"), ("src/two.py", "2\n"))], settings
    )
    agent = WorkerAgent(task=TaskModel(id="1", description="two files"), client=client)
    await agent.run()

    assert client.list_files(settings.workspace) == ["src/one.py", "src/two.py"]
    assert (_workspace / "src/one.py").read_text() == "1\n"


# --------------------------------------------------------------------------- #
# 2. the _encode_event byte cap under escape expansion
# --------------------------------------------------------------------------- #

#: Every pathological payload, as ``(name, field value)``.  Each one is applied
#: both to a single field and - in the fitting test - across the whole field
#: budget, because a single field is clipped by ``MAX_FIELD_CHARS`` before it can
#: ever reach the escaping logic.
ESCAPE_PAYLOADS: list[tuple[str, str]] = [
    ("quote_and_backslash", "x" + '"\\' * 5000),
    ("control_chars", "x" + "\n\t\r" * 2000),
    ("all_quotes", '"' * 20000),
    ("multibyte_utf8", '"' + "\u00e9\u4e2d\U0001f600" * 2000),
    # ``\n`` / ``t`` / ``r`` escape to two characters; the *other* C0 controls
    # are the real 6-byte worst case (``\u0001``), which is what can push even
    # an empty-detail envelope over the cap.
    ("six_byte_controls", "\x00\x01\x1f" * 2000),
]


def encode_with_deadline(event: dict[str, Any], seconds: float = 5.0) -> str:
    """``_encode_event`` under a hard wall-clock deadline.

    A naive incremental fitting loop on an escape-heavy payload degenerates
    towards O(n^2) serialisations; a broken one never terminates at all.  The
    ``SIGALRM`` deadline turns either failure into a test error instead of a hung
    suite, and the returned elapsed time catches the merely-slow case.
    """
    if not hasattr(signal, "setitimer"):  # pragma: no cover - non-POSIX
        pytest.skip("SIGALRM deadlines are POSIX-only")

    def on_alarm(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"_encode_event did not finish within {seconds}s")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    started = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return server._encode_event(event)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        assert time.monotonic() - started < seconds


@pytest.mark.parametrize(("name", "payload"), ESCAPE_PAYLOADS, ids=[n for n, _ in ESCAPE_PAYLOADS])
async def test_escape_heavy_frames_stay_under_the_byte_cap(name: str, payload: str) -> None:
    """Both invariants at once, on a payload that expands under ``json.dumps``."""
    frame = encode_with_deadline(wide_event(payload))

    assert len(frame.encode("utf-8")) <= server.MAX_EVENT_BYTES, (
        f"{name}: escaped frame overran the cap by "
        f"{len(frame.encode('utf-8')) - server.MAX_EVENT_BYTES} bytes"
    )
    payload_out = json.loads(frame)  # a byte slice would raise here
    assert payload_out["truncated"] is True
    assert payload_out["type"] == "task_completed"
    assert payload_out["cycle"] == 7
    # every one of these overruns the cap once clipped *and* escaped, so the
    # fitting branch ran: ``detail`` is a prefix of the clipped frame, never a
    # re-serialised field.  ``ensure_ascii`` keeps the result pure ASCII, which is
    # what makes the byte cap and the character cap agree here.
    assert frame.isascii()
    bounded = server.json.dumps(server._bound_value(wide_event(payload)), default=str)
    assert len(bounded.encode("utf-8")) > server.MAX_EVENT_BYTES
    assert bounded.startswith(payload_out["detail"])
    assert len(payload_out["detail"]) < len(bounded)


@pytest.mark.parametrize(("name", "payload"), ESCAPE_PAYLOADS, ids=[n for n, _ in ESCAPE_PAYLOADS])
async def test_escape_heavy_payloads_are_not_mutated_before_framing(
    name: str, payload: str
) -> None:
    """A single pathological field stays a valid frame too, just a clipped one.

    ``_encode_event`` has two distinct size guards, and a single assertion
    cannot cover both:

    * **clipping** - ``_bound_value`` truncates every string to
      ``MAX_FIELD_CHARS``, so most payloads are already under the cap once
      bounded, and the frame is the bounded event verbatim.
    * **fitting** - when the clipped *and escaped* event is still over
      ``MAX_EVENT_BYTES`` (``\\u0000``-style controls, and the ``ensure_ascii``
      escapes for non-ASCII, expand the most), ``_escaped_prefix`` binary-searches
      the longest prefix of the serialised frame that fits, and that prefix - not
      a field - is what lands in ``detail``.

    The detail is therefore field-length only when nothing needed fitting.
    """
    event = {"type": "log", "detail": payload, "cycle": 1}
    frame = encode_with_deadline(event)

    assert len(frame.encode("utf-8")) <= server.MAX_EVENT_BYTES
    decoded = json.loads(frame)
    assert decoded["type"] == "log"
    assert decoded["cycle"] == 1

    bounded = server.json.dumps(server._bound_value(event), default=str)
    if len(bounded.encode("utf-8")) > server.MAX_EVENT_BYTES:
        # the fitter ran: detail is a prefix of the clipped frame, not a field
        assert decoded["truncated"] is True
        assert bounded.startswith(decoded["detail"])
        assert 0 < len(decoded["detail"]) < len(bounded)
        assert frame.isascii()
    else:
        # clipping alone sufficed: the frame is the bounded event, verbatim
        assert "truncated" not in decoded
        assert decoded == json.loads(bounded)
        assert len(decoded["detail"]) <= server.MAX_FIELD_CHARS + len("\u2026[truncated]")


def test_the_post_escape_budget_is_load_bearing() -> None:
    """Why the cap is measured after escaping, demonstrated rather than asserted.

    Slicing the already-serialised frame to ``MAX_EVENT_BYTES - overhead``
    *characters* - the pre-fix approach - re-escapes every quote and backslash
    in the detail, so the frame that comes out the other end overruns the cap.
    If this test ever starts failing, the cap has become decorative again.
    """
    payload = "x" + '"\\' * 5000
    event = wide_event(payload)
    frame = server.json.dumps(server._bound_value(event), default=str)
    assert len(frame.encode("utf-8")) > server.MAX_EVENT_BYTES  # the input is oversized

    envelope = {
        "type": "task_completed",
        "cycle": 7,
        "truncated": True,
        "detail": frame[: server.MAX_EVENT_BYTES - 200],
    }
    naive = server.json.dumps(envelope, default=str)

    assert len(naive.encode("utf-8")) > server.MAX_EVENT_BYTES  # the naive slice loses
    actual = server._encode_event(event)
    assert len(actual.encode("utf-8")) <= server.MAX_EVENT_BYTES
    assert json.loads(actual)["type"] == "task_completed"


def test_pathological_type_falls_back_to_a_minimal_envelope() -> None:
    """An oversized *type* must not spin the fitter - it ships a fixed envelope.

    ``type`` is copied into the fallback envelope verbatim, and a 2000-character
    run of 6-byte control escapes is already ~12 kB, so the empty-detail
    overhead alone busts the cap.  The documented answer is a fixed
    ``{"type": "event", "truncated": True}``.
    """
    event = {"type": "\x01" * 20_000, "cycle": 2, "detail": '"' * 20_000}

    frame = encode_with_deadline(event)

    assert len(frame.encode("utf-8")) <= server.MAX_EVENT_BYTES
    assert json.loads(frame) == {"type": "event", "truncated": True}


def test_oversized_type_and_detail_stay_valid_even_without_the_minimal_fallback() -> None:
    """The same oversized-type shape that *does* fit under the cap stays intact."""
    event = {"type": '"' * 20_000, "cycle": 2, "detail": '"' * 20_000, "other": "\x01" * 20_000}

    frame = encode_with_deadline(event)

    assert len(frame.encode("utf-8")) <= server.MAX_EVENT_BYTES
    decoded = json.loads(frame)
    assert decoded["truncated"] is True
    assert decoded["cycle"] == 2
    assert len(decoded["detail"]) > 0


def test_pathological_payloads_are_framed_promptly() -> None:
    """The whole pathological set, in one go, under a single tight deadline."""
    events = [wide_event(payload) for _name, payload in ESCAPE_PAYLOADS]
    events.append({"type": "\x01" * 20_000, "detail": '"' * 20_000})

    started = time.monotonic()
    frames = [encode_with_deadline(event, seconds=5.0) for event in events]
    elapsed = time.monotonic() - started

    assert len(frames) == len(events)
    assert all(len(frame.encode("utf-8")) <= server.MAX_EVENT_BYTES for frame in frames)
    # a binary search over an escape-heavy frame is milliseconds; anything in the
    # seconds range means the fitting loop went quadratic
    assert elapsed < 2.0, f"framing {len(events)} pathological events took {elapsed:.2f}s"


def test_max_event_bytes_and_field_caps_are_unchanged() -> None:
    """The two constants the fitting arithmetic depends on."""
    assert server.MAX_EVENT_BYTES == 8192
    assert server.MAX_FIELD_CHARS == 2000
