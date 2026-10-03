from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from furrow.agents import worker as worker_module
from furrow.agents.prompts import WORKER_PROMPT
from furrow.agents.worker import (
    MAX_DEPENDENCY_RESULTS,
    MAX_DEPENDENCY_SECTION_CHARS,
    MAX_DESCRIPTION_CHARS,
    MAX_FILE_CHARS,
    MAX_PROMPT_CHARS,
    MAX_SUMMARY_CHARS,
    SECTION_DEPS,
    SECTION_LISTING,
    TEMP_PREFIX,
    TRUNCATION_MARKER,
    WorkerAgent,
    WorkerError,
    WriteRecord,
    apply_envelope_files,
    backup_run_dir,
    build_context,
    build_prompt,
    clamp_middle,
    create_backup,
    extract_edit_envelope,
    format_dependencies,
    format_report,
    format_summary,
    read_workspace_listing,
    resolve_backup_dir,
    resolve_write_path,
)
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient


class FakeClient:
    """Stand-in for ``LLMClient``: records prompts, serves fixed content, no network."""

    def __init__(
        self,
        files: dict[str, str] | None = None,
        listing: list[str] | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.files = files or {}
        self.listing = listing or []
        self.settings = Settings(workspace=workspace or Path("/tmp/furrow-worker-tests"))
        self.prompts: list[str] = []
        self.reads: list[str] = []
        self.listing_calls: list[object] = []
        self.listing_error: Exception | None = None
        self.listing_requires_directory = False
        # Separate knobs per method: a read failure must not also break ``complete``,
        # or the test cannot tell which path it is exercising.
        self.read_error: BaseException | None = None
        self.complete_error: BaseException | None = None
        self.response = "done"

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        if self.complete_error is not None:
            raise self.complete_error
        return self.response

    async def read_file(self, path: str | Path) -> str:
        self.reads.append(str(path))
        if self.read_error is not None:
            raise self.read_error
        try:
            return self.files[str(path)]
        except KeyError:
            raise OSError(f"no such file: {path}") from None

    def list_files(self, directory: str | Path | None = None) -> list[str]:
        self.listing_calls.append(directory)
        if self.listing_requires_directory and directory is None:
            raise TypeError("list_files() missing 1 required positional argument")
        if self.listing_error is not None:
            raise self.listing_error
        return list(self.listing)


def make_task(**kwargs) -> TaskModel:
    data = {"id": "1", "description": "Add a retry helper", "files": ["a.py"]}
    data.update(kwargs)
    return TaskModel(**data)


async def gather_context(client: FakeClient, task: TaskModel) -> str:
    return await build_context(task, client, client.settings.workspace)


# --- context injection -----------------------------------------------------------------


async def test_file_contents_appear_in_prompt():
    client = FakeClient(files={"a.py": "def helper():\n    return 42\n"})
    worker = WorkerAgent(task=make_task(), client=client)
    await worker.run()
    prompt = client.prompts[0]
    assert "def helper():" in prompt
    assert "return 42" in prompt
    assert "a.py" in prompt
    assert WORKER_PROMPT.strip() in prompt
    assert "Add a retry helper" in prompt


async def test_missing_file_is_noted_instead_of_raising(tmp_path):
    client = FakeClient(files={}, workspace=tmp_path)
    task = make_task(files=["gone.py"])
    context = await gather_context(client, task)
    assert "gone.py" in context
    assert "unavailable" in context
    # The task still runs and produces a prompt.
    result = await WorkerAgent(task=task, client=client).run()
    assert result == "done"


async def test_unreadable_file_error_is_caught(tmp_path):
    client = FakeClient(files={}, workspace=tmp_path)
    client.read_error = PermissionError("boom")
    task = make_task(files=["a.py"])
    context = await gather_context(client, task)
    assert "PermissionError" in context
    assert "unavailable" in context
    # Both candidate paths were tried before giving up on the file.
    assert len(client.reads) == 2


async def test_unreadable_everything_falls_back_to_listing(tmp_path):
    client = FakeClient(files={}, listing=["furrow/main.py"], workspace=tmp_path)
    task = make_task(files=["gone.py"])
    context = await gather_context(client, task)
    assert "unavailable" in context
    assert SECTION_LISTING in context
    assert "furrow/main.py" in context


async def test_cancelled_error_during_read_is_not_swallowed():
    client = FakeClient(files={})
    # CancelledError is a BaseException, so it must pass through the per-file guard and
    # reach the orchestrator's gather() rather than becoming an "unavailable" note.
    client.read_error = asyncio.CancelledError()
    task = make_task(files=["a.py"])
    with pytest.raises(asyncio.CancelledError):
        await gather_context(client, task)


# --- caps and truncation ---------------------------------------------------------------


def test_per_file_cap_is_enforced_with_marker():
    body = "x" * (MAX_FILE_CHARS * 3)
    task = make_task(files=["big.py"])
    context = asyncio.run(gather_context(FakeClient(files={"big.py": body}), task))
    assert TRUNCATION_MARKER in context
    assert body[:100] in context
    # The full body is gone, so the prompt stays inside the per-file budget.
    assert len(context) < MAX_FILE_CHARS + 2000


async def test_long_file_is_middle_truncated():
    head = "HEAD_SIGNATURE_MARKER"
    tail = "TAIL_SIGNATURE_MARKER"
    body = head + ("filler " * (MAX_FILE_CHARS * 2)) + tail
    assert len(body) > MAX_FILE_CHARS
    context = await gather_context(FakeClient(files={"big.py": body}), make_task(files=["big.py"]))
    assert head in context
    assert tail in context
    assert TRUNCATION_MARKER in context
    assert context.index(head) < context.index(TRUNCATION_MARKER) < context.index(tail)


def test_clamp_middle_keeps_both_ends_and_respects_limit():
    text = "A" * 500 + "B" * 500 + "C" * 500
    out = clamp_middle(text, 200)
    assert len(out) <= 200
    assert out.startswith("A")
    assert out.endswith("C")
    assert TRUNCATION_MARKER in out
    assert clamp_middle("short", 100) == "short"


def test_total_cap_enforced_by_prompt_assembly():
    context = "### huge.py\n" + ("z" * (MAX_PROMPT_CHARS * 2))
    prompt = build_prompt(make_task(), context)
    assert len(prompt) <= MAX_PROMPT_CHARS
    assert TRUNCATION_MARKER in prompt


def test_total_cap_enforced_for_oversized_header():
    task = make_task(description="d" * (MAX_DESCRIPTION_CHARS * 3))
    prompt = build_prompt(task, "ctx")
    assert len(prompt) <= MAX_PROMPT_CHARS
    assert len(task.description) > MAX_DESCRIPTION_CHARS


async def test_prompt_never_exceeds_total_cap(tmp_path):
    many = [f"mod{i}.py" for i in range(40)]
    files = {name: ("def f():\n" + "    return 1\n" * 2000) for name in many}
    client = FakeClient(files=files, workspace=tmp_path)
    task = make_task(files=many, description="x" * 20_000)
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    prompt = client.prompts[0]
    assert result == "done"
    assert len(prompt) <= MAX_PROMPT_CHARS
    assert TRUNCATION_MARKER in prompt


def test_description_and_paths_are_bounded():
    task = make_task(description="d" * 10_000, files=["p" * 500])
    prompt = build_prompt(task)
    assert len(prompt) <= MAX_PROMPT_CHARS
    assert "d" * 1000 in prompt


# --- workspace listing fallback ---------------------------------------------------------


async def test_empty_files_falls_back_to_workspace_listing():
    client = FakeClient(listing=["furrow/cli/main.py", "README.md"])
    context = await gather_context(client, make_task(files=[]))
    assert SECTION_LISTING in context
    assert "furrow/cli/main.py" in context
    assert "README.md" in context


async def test_raising_list_files_degrades_gracefully():
    client = FakeClient(listing=[])
    client.listing_error = OSError("permission denied")
    context = await gather_context(client, make_task(files=[]))
    assert "unavailable" in context
    result = await WorkerAgent(task=make_task(files=[]), client=client).run()
    assert result == "done"


def test_listing_uses_directory_form_when_required():
    client = FakeClient(listing=["a.py"])
    client.listing_requires_directory = True
    paths, error = read_workspace_listing(client, Path("/tmp/furrow-worker-tests"))
    assert error is None
    assert paths == ["a.py"]
    assert client.listing_calls[1] == Path("/tmp/furrow-worker-tests")


def test_listing_works_against_real_client_signature(tmp_path):
    # ``LLMClient.list_files`` takes a required ``directory`` positional argument, so the
    # no-argument probe in read_workspace_listing raises TypeError on every call. Pin the
    # workaround against the real signature so a change to llm.py cannot silently turn the
    # listing into an error note.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    client = LLMClient(settings=Settings(workspace=tmp_path))
    paths, error = read_workspace_listing(client, tmp_path)
    assert error is None
    assert paths == ["pkg/mod.py"]


def test_listing_error_is_returned_not_raised():
    client = FakeClient()
    client.listing_error = RuntimeError("nope")
    paths, error = read_workspace_listing(client, Path("/tmp"))
    assert paths == []
    assert "RuntimeError" in (error or "")


def test_workspace_listing_is_bounded():
    listing = [f"f{i}.py" for i in range(1000)]
    client = FakeClient(listing=listing)
    task = make_task(files=[])
    context = asyncio.run(gather_context(client, task))
    assert len(context) < 20_000
    assert "more files" in context


# --- dependencies ----------------------------------------------------------------------


async def test_dependency_ids_are_surfaced():
    task = make_task(dependencies=["7", "8"])
    client = FakeClient(files={"a.py": "x = 1\n"})
    worker = WorkerAgent(task=task, client=client)
    await worker.run()
    prompt = client.prompts[0]
    assert "Depends on" in prompt
    assert "7" in prompt and "8" in prompt


async def test_dependency_results_are_surfaced():
    task = make_task(dependencies=["7"])
    client = FakeClient(files={"a.py": "x = 1\n"})
    worker = WorkerAgent(
        task=task, client=client, dependency_results={"7": "created furrow/retry.py"}
    )
    await worker.run()
    assert "created furrow/retry.py" in client.prompts[0]


async def test_dependency_section_is_rendered_once():
    # The section belongs to the prompt header; build_context must not repeat it, or a long
    # dependency chain would eat the single MAX_PROMPT_CHARS budget twice over.
    client = FakeClient(files={"a.py": "x = 1\n"})
    agent = WorkerAgent(
        task=make_task(dependencies=["7"]),
        client=client,
        dependency_results={"7": "created furrow/retry.py"},
    )
    await agent.run()
    prompt = client.prompts[0]
    assert prompt.count(SECTION_DEPS) == 1
    assert prompt.count("created furrow/retry.py") == 1


def test_format_dependencies_is_bounded():
    task = make_task(dependencies=[str(i) for i in range(100)])
    out = format_dependencies(task, {str(i): "r" * 5_000 for i in range(100)})
    assert len(out) <= MAX_DEPENDENCY_SECTION_CHARS
    # The overflow notices survive the head cap even though the result bodies do not.
    assert "further dependencies" in out
    assert "further results" in out
    assert out.count("- [task ") <= MAX_DEPENDENCY_RESULTS


def test_format_dependencies_empty_when_no_dependencies():
    assert format_dependencies(make_task(), None) == ""


# --- run() behaviour ---------------------------------------------------------------------


async def test_run_returns_model_text():
    client = FakeClient(files={"a.py": "x = 1\n"})
    client.response = "Changed a.py: added helper.\nComplete: yes"
    result = await WorkerAgent(task=make_task(), client=client).run()
    assert result == client.response
    assert client.prompts[0].count("x = 1") == 1


async def test_run_uses_worker_model_from_settings():
    client = FakeClient(files={"a.py": "x = 1\n"})
    client.settings = Settings(worker_model="custom-worker-model", workspace=Path("/tmp"))
    agent = WorkerAgent(task=make_task(), client=client)
    assert agent.settings is client.settings
    assert agent.settings.worker_model == "custom-worker-model"


async def test_client_error_becomes_clear_worker_error():
    client = FakeClient(files={"a.py": "x = 1\n"})
    client.complete_error = RuntimeError("upstream 503")
    with pytest.raises(WorkerError) as excinfo:
        await WorkerAgent(task=make_task(), client=client).run()
    message = str(excinfo.value)
    assert "1" in message
    assert "upstream 503" in message
    assert "RuntimeError" in message
    assert isinstance(excinfo.value, RuntimeError)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


async def test_cancelled_error_is_not_swallowed():
    client = FakeClient(files={"a.py": "x = 1\n"})
    client.complete_error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await WorkerAgent(task=make_task(), client=client).run()


def test_summary_is_bounded_and_defaulted():
    assert format_summary("  ok  ") == "ok"
    assert "no output" in format_summary("")
    assert "no output" in format_summary("   \n ")
    long_summary = "s" * (MAX_SUMMARY_CHARS * 2)
    bounded = format_summary(long_summary)
    assert len(bounded) <= MAX_SUMMARY_CHARS
    assert TRUNCATION_MARKER in bounded


def test_orchestrator_style_construction_without_client_kwargs():
    # Mirrors the existing call site: keyword-only task/client, no settings, no deps.
    client = FakeClient(files={"a.py": "x = 1\n"})
    agent = WorkerAgent(task=make_task(), client=client)
    assert agent.task.id == "1"
    assert agent.client is client
    assert agent.settings.worker_model


async def test_run_accepts_dependency_results_per_call():
    client = FakeClient(files={"a.py": "x = 1\n"})
    agent = WorkerAgent(task=make_task(dependencies=["7"]), client=client)
    assert await agent.run() == "done"
    assert "Depends on" in client.prompts[0]
    # The call-time mapping wins for that call only; the agent keeps its own state.
    await agent.run(dependency_results={"7": "added furrow/retry.py"})
    assert "added furrow/retry.py" in client.prompts[1]
    assert agent.dependency_results == {}
    await agent.run()
    assert "added furrow/retry.py" not in client.prompts[2]


# --- edit protocol: envelope extraction --------------------------------------------------
#
# Everything below is hermetic: the LLM is a FakeClient returning canned replies, and the
# only paths written are inside tmp_path.


def envelope(
    files: list[dict] | None = None,
    summary: str = "did the thing",
    complete: bool = True,
    issues: list[str] | None = None,
) -> str:
    """A canned worker reply in the protocol WORKER_PROMPT asks for."""
    return json.dumps(
        {
            "summary": summary,
            "complete": complete,
            "issues": issues or [],
            "files": files or [],
        }
    )


def write_edit(path: str, content: str) -> dict:
    return {"path": path, "action": "write", "content": content}


def make_worker(tmp_path: Path, response: str, **settings_kwargs) -> WorkerAgent:
    """A worker whose settings point every write at ``tmp_path``."""
    client = FakeClient(files={}, workspace=tmp_path)
    settings_kwargs.setdefault("backup_dir", tmp_path / ".furrow/backups")
    client.settings = Settings(workspace=tmp_path, **settings_kwargs)
    client.response = response
    return WorkerAgent(task=make_task(files=[]), client=client)


def only(records: list[WriteRecord]) -> WriteRecord:
    assert len(records) == 1, [r.reason for r in records]
    return records[0]


def test_envelope_parses_when_fenced_in_json_block():
    text = "```json\n" + envelope([write_edit("a.py", "x = 1\n")]) + "\n```"
    payload = extract_edit_envelope(text)
    assert payload is not None
    assert payload["summary"] == "did the thing"
    assert payload["files"][0]["path"] == "a.py"


def test_envelope_parses_when_wrapped_in_prose():
    text = (
        "Sure - here is the change you asked for:\n"
        '{"summary": "patched", "complete": true, "issues": [], '
        '"files": [{"path": "a.py", "action": "write", "content": "x = 1\\n"}]}\n'
        "Hope that helps."
    )
    payload = extract_edit_envelope(text)
    assert payload is not None
    assert payload["files"][0]["content"] == "x = 1\n"


def test_envelope_extraction_skips_a_decoy_object():
    text = (
        '{"example": {"path": "nope.py"}}\n'
        "The real change follows.\n"
        '{"summary": "ok", "files": [{"path": "yes.py", "action": "write", "content": "y\\n"}]}'
    )
    payload = extract_edit_envelope(text)
    assert payload is not None
    assert payload["files"][0]["path"] == "yes.py"


def test_envelope_extraction_returns_none_for_prose():
    assert extract_edit_envelope("The task is already done, nothing to change.") is None
    assert extract_edit_envelope("") is None
    assert extract_edit_envelope("   \n") is None
    assert extract_edit_envelope(None) is None


def test_envelope_extraction_never_repairs_invalid_json():
    # A trailing comma makes the object unparseable; rewriting it would write whatever
    # the repairer guessed, so the reply is treated as prose instead.
    assert extract_edit_envelope('{"summary": "x", "files": [],}') is None


def test_envelope_extraction_accepts_a_keys_only_object():
    payload = extract_edit_envelope('{"complete": true, "issues": ["left TODO"]}')
    assert payload == {"complete": True, "issues": ["left TODO"]}


async def test_prose_reply_applies_nothing_and_is_returned_as_the_summary(tmp_path):
    reply = "Nothing to do: the helper already exists."
    worker = make_worker(tmp_path, reply)
    result = await worker.run()
    assert result == reply
    assert worker.advisory is True
    assert list(tmp_path.iterdir()) == []
    record = only(worker.applied)
    assert record.action == "skipped"
    assert "envelope" in record.reason


async def test_valid_write_creates_file_with_exactly_that_content(tmp_path):
    content = 'def helper():\n    return 42\n'
    worker = make_worker(tmp_path, envelope([write_edit("pkg/helper.py", content)]))
    result = await worker.run()

    written = tmp_path / "pkg/helper.py"
    assert written.read_text(encoding="utf-8") == content
    record = only(worker.applied)
    assert (record.path, record.action, record.bytes) == (
        "pkg/helper.py",
        "created",
        len(content.encode("utf-8")),
    )
    assert record.backup_path is None  # nothing existed, so nothing to back up
    assert "created pkg/helper.py" in result


async def test_overwrite_records_modified_and_backup_holds_the_original(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("ORIGINAL = 1\n", encoding="utf-8")
    worker = make_worker(tmp_path, envelope([write_edit("a.py", "NEW = 2\n")]))
    result = await worker.run()

    assert target.read_text(encoding="utf-8") == "NEW = 2\n"
    record = only(worker.applied)
    assert record.action == "modified"
    assert record.backup_path is not None

    backup = Path(record.backup_path)
    assert backup.is_file()
    # The backup is the file as it was, and it restores cleanly.
    assert backup.read_text(encoding="utf-8") == "ORIGINAL = 1\n"
    # <workspace>/.furrow/backups/<UTC stamp>-<task id>/<relative path>
    assert backup.parent.parent == tmp_path / ".furrow/backups"
    assert backup.name == "a.py"
    assert backup.parent.name.endswith("-1")
    target.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "ORIGINAL = 1\n"
    assert "modified a.py" in result


async def test_a_second_backup_of_the_same_path_is_not_clobbered(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("first\n", encoding="utf-8")
    first = create_backup(target, "a.py", tmp_path / "backups")
    target.write_text("second\n", encoding="utf-8")
    second = create_backup(target, "a.py", tmp_path / "backups")
    assert first != second
    assert first.read_text(encoding="utf-8") == "first\n"
    assert second.read_text(encoding="utf-8") == "second\n"


async def test_overwrite_is_refused_when_the_backup_cannot_be_created(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("ORIGINAL = 1\n", encoding="utf-8")
    # A plain file where the backup directory should go: mkdir fails with OSError.
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory\n", encoding="utf-8")
    worker = make_worker(tmp_path, envelope([write_edit("a.py", "NEW = 2\n")]),
                         backup_dir=blocker / "backups")
    result = await worker.run()

    assert target.read_text(encoding="utf-8") == "ORIGINAL = 1\n"  # untouched
    record = only(worker.applied)
    assert record.action == "skipped"
    assert "backup failed" in record.reason
    assert "a.py" in result


async def test_atomic_write_leaves_no_temp_file_behind(tmp_path):
    worker = make_worker(tmp_path, envelope([write_edit("a.py", "x = 1\n")]))
    await worker.run()
    assert list(tmp_path.rglob(f"{TEMP_PREFIX}*")) == []
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
async def test_written_files_keep_a_sensible_mode(tmp_path):
    script = tmp_path / "build.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o755)

    worker = make_worker(
        tmp_path,
        envelope([write_edit("build.sh", "#!/bin/sh\necho hi\n"), write_edit("a.py", "x\n")]),
    )
    await worker.run()

    # An overwritten executable keeps its bits; a new file is not left private.
    assert script.stat().st_mode & 0o777 == 0o755
    assert (tmp_path / "a.py").stat().st_mode & 0o044 == 0o044


async def test_failed_write_leaves_no_partial_file_and_no_temp_file(tmp_path, monkeypatch):
    target = tmp_path / "a.py"
    target.write_text("ORIGINAL = 1\n", encoding="utf-8")
    replacement = "NEW = 2\n" * 5000

    def boom(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(os, "replace", boom)
    worker = make_worker(tmp_path, envelope([write_edit("a.py", replacement)]))
    result = await worker.run()

    assert target.read_text(encoding="utf-8") == "ORIGINAL = 1\n"  # never half-written
    assert list(tmp_path.rglob(f"{TEMP_PREFIX}*")) == []
    record = only(worker.applied)
    assert record.action == "skipped"
    assert "write failed" in record.reason
    assert "write failed" in result


@pytest.mark.parametrize(
    "path",
    [
        "../evil.py",
        "../../etc/passwd",
        "a/../../b.py",
        "pkg/../../../outside.py",
        "/etc/passwd",
        "~/escape.py",
        "C:\\windows\\evil.py",
        "",
        "   ",
    ],
)
async def test_paths_that_escape_the_workspace_are_rejected(tmp_path, path):
    worker = make_worker(tmp_path, envelope([write_edit(path, "owned = True\n")]))
    result = await worker.run()

    assert list(tmp_path.iterdir()) == []
    record = only(worker.applied)
    assert record.action == "skipped"
    assert record.reason
    assert str(path).strip()[:20] in result or "files[0]" in result


async def test_symlink_pointing_outside_the_workspace_is_rejected(tmp_path):
    outside = tmp_path.parent / "outside-target"
    outside.mkdir()
    secret = outside / "secret.py"
    secret.write_text("SECRET = 1\n", encoding="utf-8")
    link = tmp_path / "link.py"
    os.symlink(secret, link)

    worker = make_worker(tmp_path, envelope([write_edit("link.py", "pwned = True\n")]))
    result = await worker.run()

    assert secret.read_text(encoding="utf-8") == "SECRET = 1\n"
    record = only(worker.applied)
    assert record.action == "skipped"
    assert "outside the workspace" in record.reason
    assert "link.py" in result


@pytest.mark.parametrize(
    "path",
    [
        ".git/config",
        ".venv/x.py",
        "node_modules/pkg/index.js",
        "__pycache__/x.py",
        "src/__pycache__/x.py",
        "dist/x.py",
        "build/x.py",
        "furrow.egg-info/PKG-INFO",
        "foo.pyc",
        "pkg/foo.pyo",
    ],
)
async def test_protected_and_generated_paths_are_rejected(tmp_path, path):
    worker = make_worker(tmp_path, envelope([write_edit(path, "owned = True\n")]))
    await worker.run()

    record = only(worker.applied)
    assert record.action == "skipped"
    assert record.reason
    assert not (tmp_path / path).exists()
    assert list(tmp_path.iterdir()) == []


async def test_existing_directory_is_rejected(tmp_path):
    (tmp_path / "pkg").mkdir()
    worker = make_worker(tmp_path, envelope([write_edit("pkg", "x = 1\n")]))
    await worker.run()
    assert only(worker.applied).action == "skipped"
    assert "directory" in only(worker.applied).reason
    assert (tmp_path / "pkg").is_dir()


async def test_per_file_size_cap_is_enforced(tmp_path):
    worker = make_worker(tmp_path, envelope([write_edit("big.py", "x" * 5_000)]),
                         max_write_bytes=1_000)
    result = await worker.run()
    assert not (tmp_path / "big.py").exists()
    record = only(worker.applied)
    assert record.action == "skipped"
    assert "max_write_bytes" in record.reason
    assert "max_write_bytes" in result


async def test_total_size_cap_is_enforced(tmp_path):
    reply = envelope([write_edit("a.py", "a" * 600), write_edit("b.py", "b" * 600)])
    worker = make_worker(tmp_path, reply, max_total_write_bytes=1_000)
    await worker.run()

    assert (tmp_path / "a.py").exists()  # the first write fits
    assert not (tmp_path / "b.py").exists()
    rejected = [r for r in worker.applied if r.action == "skipped"]
    assert len(rejected) == 1
    assert "max_total_write_bytes" in rejected[0].reason


async def test_file_count_cap_is_enforced(tmp_path):
    reply = envelope([write_edit(f"m{i}.py", f"x = {i}\n") for i in range(5)])
    worker = make_worker(tmp_path, reply, max_files_per_task=2)
    await worker.run()

    assert sorted(p.name for p in tmp_path.glob("*.py")) == ["m0.py", "m1.py"]
    rejected = [r for r in worker.applied if r.action == "skipped"]
    assert len(rejected) == 3
    assert all("max_files_per_task" in r.reason for r in rejected)


async def test_non_string_content_is_rejected(tmp_path):
    reply = json.dumps(
        {"summary": "s", "files": [{"path": "a.py", "action": "write", "content": 42}]}
    )
    worker = make_worker(tmp_path, reply)
    await worker.run()
    assert not (tmp_path / "a.py").exists()
    assert "content" in only(worker.applied).reason


@pytest.mark.parametrize("action", ["patch", "delete", "", "diff", None, 3])
async def test_actions_other_than_write_are_rejected(tmp_path, action):
    reply = json.dumps(
        {"summary": "s", "files": [{"path": "a.py", "action": action, "content": "x = 1\n"}]}
    )
    worker = make_worker(tmp_path, reply)
    await worker.run()
    assert not (tmp_path / "a.py").exists()
    assert "action" in only(worker.applied).reason


async def test_malformed_files_entry_is_rejected_without_losing_the_good_ones(tmp_path):
    reply = json.dumps(
        {
            "summary": "s",
            "files": [
                "not an object",
                {"path": "good.py", "action": "write", "content": "x = 1\n"},
                {"action": "write", "content": "y = 2\n"},
            ],
        }
    )
    worker = make_worker(tmp_path, reply)
    result = await worker.run()
    assert (tmp_path / "good.py").read_text(encoding="utf-8") == "x = 1\n"
    actions = [r.action for r in worker.applied]
    assert actions == ["skipped", "created", "skipped"]
    assert "Rejected or skipped 2 requested edit(s)" in result


async def test_apply_changes_false_applies_nothing(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("ORIGINAL = 1\n", encoding="utf-8")
    worker = make_worker(
        tmp_path,
        envelope([write_edit("a.py", "NEW = 2\n"), write_edit("new.py", "n = 1\n")]),
        apply_changes=False,
    )
    result = await worker.run()

    assert target.read_text(encoding="utf-8") == "ORIGINAL = 1\n"
    assert not (tmp_path / "new.py").exists()
    assert list(tmp_path.iterdir()) == [target]  # no backups either
    assert {r.action for r in worker.applied} == {"skipped"}
    assert all("apply_changes" in r.reason for r in worker.applied)
    assert "Applied 0 file(s)" not in result


async def test_summary_reports_both_applied_and_rejected_files(tmp_path):
    reply = envelope(
        [write_edit("good.py", "x = 1\n"), write_edit("../evil.py", "x = 2\n")],
        summary="added the helper",
        issues=["could not cover the retry branch"],
    )
    worker = make_worker(tmp_path, reply)
    result = await worker.run()

    assert "added the helper" in result
    assert "good.py" in result
    assert "evil.py" in result
    assert "could not cover the retry branch" in result
    assert "Worker marked the task complete: yes" in result
    assert len(result) <= MAX_SUMMARY_CHARS


async def test_records_are_rebuilt_on_every_run(tmp_path):
    worker = make_worker(tmp_path, envelope([write_edit("a.py", "x = 1\n")]))
    await worker.run()
    assert [r.action for r in worker.applied] == ["created"]
    worker.client.response = "no change needed"
    await worker.run()
    assert [r.action for r in worker.applied] == ["skipped"]
    assert worker.advisory is True


async def test_all_new_settings_have_usable_defaults():
    settings = Settings()
    assert settings.apply_changes is True
    assert settings.backup_dir == Path(".furrow/backups")
    assert settings.max_files_per_task == 10
    assert settings.max_write_bytes == 200_000
    assert settings.max_total_write_bytes == 1_000_000


def test_advisory_mode_is_selectable_from_the_environment(monkeypatch):
    monkeypatch.setenv("FURROW_APPLY_CHANGES", "0")
    assert Settings().apply_changes is False
    monkeypatch.delenv("FURROW_APPLY_CHANGES")
    assert Settings().apply_changes is True


def test_apply_envelope_files_is_usable_without_the_agent(tmp_path):
    payload = {"summary": "s", "files": [write_edit("a.py", "x = 1\n")]}
    settings = Settings(workspace=tmp_path, backup_dir=tmp_path / ".furrow/backups")
    records = asyncio.run(
        apply_envelope_files(payload, workspace=tmp_path, settings=settings, task_id="7")
    )
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\n"
    assert only(records).action == "created"


def test_files_must_be_an_array(tmp_path):
    payload = {"summary": "s", "files": "a.py"}
    settings = Settings(workspace=tmp_path)
    records = asyncio.run(
        apply_envelope_files(payload, workspace=tmp_path, settings=settings, task_id="1")
    )
    assert only(records).action == "skipped"
    assert "JSON array" in only(records).reason


def test_backup_paths_are_anchored_in_the_workspace(tmp_path):
    assert resolve_backup_dir(Path(".furrow/backups"), tmp_path) == tmp_path / ".furrow/backups"
    absolute = tmp_path / "elsewhere"
    assert resolve_backup_dir(absolute, tmp_path) == absolute
    run_dir = backup_run_dir(tmp_path / ".furrow/backups", "1/../../weird")
    # The task id cannot contribute a path segment of its own.
    assert "/" not in run_dir.name
    assert run_dir.name.endswith("weird")
    assert run_dir.parent == tmp_path / ".furrow/backups"


def test_resolve_write_path_returns_an_absolute_destination(tmp_path):
    target = resolve_write_path("pkg/a.py", tmp_path)
    assert target == tmp_path / "pkg/a.py"
    # Even a '..' that would resolve back inside the workspace is refused: the model
    # should not be guessing at paths it was not given.
    for bad in (
        "pkg/../a.py",
        "../evil.py",
        "/etc/passwd",
        ".git/config",
        "foo.pyc",
        "",
        7,
        None,
    ):
        with pytest.raises(ValueError):
            resolve_write_path(bad, tmp_path)


def test_format_report_keeps_rejections_visible_when_the_summary_is_huge():
    records = [WriteRecord(path=f"m{i}.py", action="created", bytes=1) for i in range(10)]
    records.append(WriteRecord(path="../evil.py", action="skipped", reason="escapes"))
    report = format_report("s" * 20_000, records, issues=["i" * 2_000])
    assert len(report) <= MAX_SUMMARY_CHARS
    assert "../evil.py" in report
    assert "m0.py" in report
    assert TRUNCATION_MARKER in report


def test_testresult_is_not_collected_by_pytest():
    # pytest warns about any class named Test* with an __init__; the shipped model is
    # named TestResult, so it must opt out without becoming a pydantic field.
    assert getattr(TestResult, "__test__", True) is False
    result = TestResult(passed=True, summary="ok")
    assert (result.passed, result.summary, result.failures) == (True, "ok", [])
    assert list(TestResult.model_fields) == ["passed", "summary", "failures"]


def test_worker_prompt_states_the_edit_contract():
    assert '"files"' in WORKER_PROMPT
    assert '"action"' in WORKER_PROMPT
    assert "raw JSON only" in WORKER_PROMPT
    assert "COMPLETE new content" in WORKER_PROMPT
    assert worker_module.ACTION_WRITE == "write"
