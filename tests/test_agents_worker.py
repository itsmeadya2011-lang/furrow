"""Regression tests for :class:`furrow.agents.worker.WorkerAgent`.

Load-bearing regressions pinned here:

* the worker actually **writes files into the workspace** from the model's
  structured JSON reply, and ``run()`` returns a ``str``;
* **path escape is refused** - a reply asking for ``../escape.py`` or
  ``/tmp/escape.py`` writes nothing outside ``settings.workspace``;
* a missing or unreadable pre-existing file does not crash the worker;
* the worker uses ``settings.worker_model``.

Every assertion about "nothing was written" is made against ``tmp_path`` (or an
absolute path that provably does not exist before the test), so a failing
escape-check cannot litter the repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient
from tests.conftest import WORKER, ScriptedCompletion, install_scripted, make_settings, worker_reply


def build_worker(settings: Settings, worker_task: TaskModel | None = None) -> WorkerAgent:
    """A worker bound to *settings*, using a throwaway task by default."""
    return WorkerAgent(
        task=worker_task or TaskModel(id="1", description="d"),
        client=LLMClient(settings=settings),
    )


def make_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reply: str | Any, **setting_overrides: Any
) -> tuple[WorkerAgent, ScriptedCompletion, Settings]:
    """A worker wired to an isolated workspace whose LLM answers with *reply*."""
    settings = make_settings(tmp_path / "ws", **setting_overrides)
    scripted = install_scripted(monkeypatch, worker=reply)
    agent = WorkerAgent(
        task=TaskModel(id="1", description="do"), client=LLMClient(settings=settings)
    )
    return agent, scripted, settings


# ---------------------------------------------------------------- happy paths


async def test_worker_writes_files_into_the_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path / "ws")
    scripted = install_scripted(
        monkeypatch,
        worker=worker_reply(
            summary="added the module",
            files=[
                {"path": "pkg/mod.py", "content": "VALUE = 1\n"},
                {"path": "tests/test_mod.py", "content": "def test_x():\n    assert True\n"},
            ],
        ),
    )
    worker = WorkerAgent(
        task=TaskModel(id="1", description="add a module"), client=LLMClient(settings=settings)
    )
    result = await worker.run()
    workspace = tmp_path / "ws"
    assert (workspace / "pkg" / "mod.py").read_text() == "VALUE = 1\n"
    assert (workspace / "tests" / "test_mod.py").exists()
    assert isinstance(result, str)
    assert "pkg/mod.py" in result
    assert "tests/test_mod.py" in result
    assert "added the module" in result
    assert scripted.models(WORKER) == [settings.worker_model]


async def test_worker_returns_a_string_even_without_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker, _, _ = make_worker(monkeypatch, tmp_path, worker_reply(summary="nothing to do"))
    result = await worker.run()
    assert isinstance(result, str)
    assert "No files written" in result
    assert "nothing to do" in result


async def test_worker_creates_a_missing_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "not-there-yet"
    settings = make_settings(workspace)
    install_scripted(monkeypatch, worker=worker_reply(files=[{"path": "a.py", "content": "a\n"}]))
    await build_worker(settings).run()
    assert (workspace / "a.py").read_text() == "a\n"


async def test_worker_overwrites_an_existing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "a.py").write_text("old")
    install_scripted(monkeypatch, worker=worker_reply(files=[{"path": "a.py", "content": "new\n"}]))
    await build_worker(settings).run()
    assert (tmp_path / "ws" / "a.py").read_text() == "new\n"


async def test_worker_skips_malformed_file_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reply = json.dumps(
        {
            "summary": "partial",
            "files": [
                "just a string",
                {"content": "no path"},
                {"path": "ok.py", "content": 123},
                {"path": "  ", "content": "blank path"},
                {"path": "good.py", "content": "good\n"},
            ],
        }
    )
    worker, _, _ = make_worker(monkeypatch, tmp_path, reply)
    result = await worker.run()
    assert (tmp_path / "ws" / "good.py").read_text() == "good\n"
    assert "good.py" in result
    assert "just a string" not in result


# ------------------------------------------------------------- path safety


@pytest.mark.parametrize(
    "escaping_path",
    [
        "../escape.py",
        "nested/../../escape.py",
        "./../escape.py",
        "a/b/../../../escape.py",
    ],
)
async def test_worker_refuses_relative_path_escape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, escaping_path: str
) -> None:
    worker, _, _ = make_worker(
        monkeypatch, tmp_path, worker_reply(files=[{"path": escaping_path, "content": "PWNED\n"}])
    )
    result = await worker.run()
    assert not (tmp_path / "escape.py").exists(), f"{escaping_path} escaped the workspace"
    assert list(tmp_path.iterdir()) == [tmp_path / "ws"], (
        "nothing may be created outside the workspace"
    )
    assert "Rejected (path escape)" in result


async def test_worker_refuses_an_absolute_path_outside_the_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = Path("/tmp") / f"furrow-escape-{tmp_path.name}.py"
    assert not target.exists()
    worker, _, _ = make_worker(
        monkeypatch, tmp_path, worker_reply(files=[{"path": str(target), "content": "PWNED\n"}])
    )
    result = await worker.run()
    assert not target.exists()
    assert "Rejected (path escape)" in result


async def test_worker_writes_the_safe_files_of_a_mixed_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker, _, _ = make_worker(
        monkeypatch,
        tmp_path,
        worker_reply(
            summary="half the job",
            files=[
                {"path": "safe.py", "content": "SAFE = True\n"},
                {"path": "../unsafe.py", "content": "UNSAFE = True\n"},
            ],
        ),
    )
    result = await worker.run()
    assert (tmp_path / "ws" / "safe.py").read_text() == "SAFE = True\n"
    assert not (tmp_path / "unsafe.py").exists()
    assert "safe.py" in result
    assert "../unsafe.py" in result
    assert "Rejected (path escape)" in result


async def test_worker_accepts_a_dotdot_path_that_stays_inside(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``a/../b.py`` resolves inside the workspace and must be allowed."""
    worker, _, _ = make_worker(
        monkeypatch, tmp_path, worker_reply(files=[{"path": "a/../inside.py", "content": "ok\n"}])
    )
    result = await worker.run()
    assert (tmp_path / "ws" / "inside.py").read_text() == "ok\n"
    assert "Rejected" not in result


async def test_worker_accepts_an_absolute_path_inside_the_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    target = tmp_path / "ws" / "abs.py"
    install_scripted(
        monkeypatch,
        worker=worker_reply(files=[{"path": str(target), "content": "a\n"}]),
    )
    await build_worker(settings).run()
    assert (tmp_path / "ws" / "abs.py").read_text() == "a\n"


# ------------------------------------------------------------- context


async def test_worker_includes_the_workspace_listing_in_the_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "existing.py").write_text("x = 1\n")
    scripted = install_scripted(monkeypatch, worker=worker_reply())
    await build_worker(settings).run()
    prompt = scripted.only_prompt(WORKER)
    assert "existing.py" in prompt


async def test_worker_includes_the_contents_of_the_declared_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "target.py").write_text("SECRET_MARKER = 1\n")
    scripted = install_scripted(monkeypatch, worker=worker_reply())
    task = TaskModel(id="1", description="d", files=["target.py"])
    await WorkerAgent(task=task, client=LLMClient(settings=settings)).run()
    assert "SECRET_MARKER" in scripted.only_prompt(WORKER)


async def test_worker_truncates_a_huge_file_in_the_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "big.py").write_text("HEAD_MARKER\n" + ("filler\n" * 4000) + "TAIL_MARKER\n")
    scripted = install_scripted(monkeypatch, worker=worker_reply())
    task = TaskModel(id="1", description="d", files=["big.py"])
    await WorkerAgent(task=task, client=LLMClient(settings=settings)).run()
    prompt = scripted.only_prompt(WORKER)
    assert "chars truncated" in prompt
    assert "HEAD_MARKER" in prompt
    assert len(prompt) < 40000


async def test_worker_names_the_files_it_may_touch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path / "ws")
    scripted = install_scripted(monkeypatch, worker=worker_reply())
    task = TaskModel(id="1", description="d", files=["a.py", "b.py"])
    await WorkerAgent(task=task, client=LLMClient(settings=settings)).run()
    assert "a.py, b.py" in scripted.only_prompt(WORKER)


# ----------------------------------------------------------- robustness


async def test_worker_ignores_a_missing_pre_existing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    install_scripted(monkeypatch, worker=worker_reply(summary="fine"))
    task = TaskModel(id="1", description="d", files=["not-here.py", "also-missing.py"])
    result = await WorkerAgent(task=task, client=LLMClient(settings=settings)).run()
    assert "fine" in result


async def test_worker_ignores_a_directory_named_as_a_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path / "ws")
    (tmp_path / "ws" / "pkg").mkdir(parents=True)
    install_scripted(monkeypatch, worker=worker_reply(summary="fine"))
    task = TaskModel(id="1", description="d", files=["pkg"])
    result = await WorkerAgent(task=task, client=LLMClient(settings=settings)).run()
    assert "fine" in result


async def test_worker_survives_an_unreadable_pre_existing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``read_file`` raising ``OSError`` must be logged, not propagated."""
    settings = make_settings(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "locked.py").write_text("secret")
    client = LLMClient(settings=settings)

    async def unreadable(path: Path) -> str:
        raise OSError("permission denied")

    client.read_file = unreadable  # type: ignore[method-assign]
    install_scripted(monkeypatch, worker=worker_reply(summary="still fine"))
    task = TaskModel(id="1", description="d", files=["locked.py"])
    result = await WorkerAgent(task=task, client=client).run()
    assert "still fine" in result


async def test_worker_reports_a_non_json_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker, _, _ = make_worker(monkeypatch, tmp_path, "I wrote the file already!")
    result = await worker.run()
    assert isinstance(result, str)
    assert "failed to parse response as JSON" in result
    assert list((tmp_path / "ws").iterdir()) == []


async def test_worker_uses_the_worker_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path / "ws", worker_model="worker-model-x")
    scripted = install_scripted(monkeypatch, worker=worker_reply())
    await build_worker(settings).run()
    assert scripted.models(WORKER) == ["worker-model-x"]


async def test_worker_falls_back_to_the_settings_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``WorkerAgent(task=...)`` builds its own client from ``settings=None``."""
    from furrow.config import settings as global_settings

    install_scripted(monkeypatch, worker=worker_reply(summary="singleton"))
    worker = WorkerAgent(task=TaskModel(id="1", description="d"))
    assert worker.client.settings is global_settings
