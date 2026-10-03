"""Unit tests for ``FILE:`` block parsing and path sandboxing in the worker agent."""

from __future__ import annotations

from pathlib import Path

from furrow.agents.worker import WorkerAgent, _safe_path, parse_file_blocks
from furrow.config import Settings, TaskModel

# --------------------------------------------------------------------------- #
# _safe_path
# --------------------------------------------------------------------------- #


def test_safe_path_rejects_an_absolute_path(tmp_path):
    assert _safe_path("/etc/passwd", tmp_path.resolve()) is None


def test_safe_path_rejects_a_parent_escape(tmp_path):
    assert _safe_path("../escape.py", tmp_path.resolve()) is None


def test_safe_path_rejects_an_empty_path(tmp_path):
    assert _safe_path("", tmp_path.resolve()) is None


def test_safe_path_rejects_an_interior_parent_escape(tmp_path):
    assert _safe_path("sub/../../escape.py", tmp_path.resolve()) is None


def test_safe_path_accepts_a_simple_relative_path(tmp_path):
    root = tmp_path.resolve()
    assert _safe_path("a.py", root) == root / "a.py"


def test_safe_path_accepts_a_nested_relative_path(tmp_path):
    root = tmp_path.resolve()
    assert _safe_path("nested/dir/b.py", root) == root / "nested" / "dir" / "b.py"


def test_safe_path_strips_surrounding_whitespace_and_backticks(tmp_path):
    root = tmp_path.resolve()
    assert _safe_path("  a.py  ", root) == root / "a.py"
    assert _safe_path("`a.py`", root) == root / "a.py"


def test_safe_path_rejects_a_dot_path(tmp_path):
    # "." resolves to the root itself, which is not a file the worker may write.
    assert _safe_path(".", tmp_path.resolve()) is None


# --------------------------------------------------------------------------- #
# parse_file_blocks
# --------------------------------------------------------------------------- #


def test_parse_file_blocks_single_block():
    text = "Here you go.\n\nFILE: a.py\n```python\nprint(1)\n```\n"
    assert parse_file_blocks(text) == [("a.py", "print(1)\n")]


def test_parse_file_blocks_two_blocks_in_order():
    text = "FILE: a.py\n```\nalpha\n```\nFILE: sub/b.py\n```\nbeta\ngamma\n```\n"
    assert parse_file_blocks(text) == [
        ("a.py", "alpha\n"),
        ("sub/b.py", "beta\ngamma\n"),
    ]


def test_parse_file_blocks_language_tagged_fence():
    text = "FILE: main.py\n```python\ndef f():\n    return 1\n```\n"
    assert parse_file_blocks(text) == [("main.py", "def f():\n    return 1\n")]


def test_parse_file_blocks_empty_body():
    text = "FILE: empty.py\n```\n```\n"
    assert parse_file_blocks(text) == [("empty.py", "")]


def test_parse_file_blocks_header_is_case_insensitive_and_may_be_indented():
    text = "  file: a.py  \n```\nx = 1\n```\n"
    assert parse_file_blocks(text) == [("a.py", "x = 1\n")]


def test_parse_file_blocks_prose_without_a_header_yields_nothing():
    text = "I could not find anything to write.\nUse FILE: headers next time.\n"
    assert parse_file_blocks(text) == []


def test_parse_file_blocks_fence_without_a_header_yields_nothing():
    assert parse_file_blocks("```python\nprint(1)\n```\n") == []


def test_parse_file_blocks_unterminated_fence_yields_nothing():
    assert parse_file_blocks("FILE: a.py\n```python\nprint(1)\n") == []


# --------------------------------------------------------------------------- #
# WorkerAgent.run
# --------------------------------------------------------------------------- #


class _FakeClient:
    """Records the writes the worker asks for, and performs them for real."""

    def __init__(self, response: str, settings: Settings) -> None:
        self.response = response
        self.settings = settings
        self.writes: list[tuple[str, str]] = []
        self.prompts: list[tuple[str, str | None]] = []

    async def complete(self, prompt, system: str = "", model: str | None = None) -> str:
        self.prompts.append((prompt, model))
        return self.response

    async def write_file(self, path, content) -> None:
        self.writes.append((str(path), content))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def _worker(tmp_path, response: str) -> tuple[WorkerAgent, _FakeClient]:
    # The workspace MUST be tmp_path: a real WorkerAgent writes wherever it is told.
    settings = Settings(_env_file=None, workspace=tmp_path)
    client = _FakeClient(response, settings)
    task = TaskModel(id="1", description="do the thing")
    return WorkerAgent(task, client=client, settings=settings), client


async def test_run_writes_every_block_and_reports_relative_paths(tmp_path):
    response = (
        "Sure, here you go.\n\n"
        "FILE: a.py\n```python\nprint('a')\n```\n\n"
        "FILE: sub/b.py\n```python\nprint('b')\n```\n"
    )
    agent, client = _worker(tmp_path, response)

    summary = await agent.run()

    assert (tmp_path / "a.py").read_text() == "print('a')\n"
    assert (tmp_path / "sub" / "b.py").read_text() == "print('b')\n"
    assert [content for _, content in client.writes] == ["print('a')\n", "print('b')\n"]

    written = [Path(path) for path, _ in client.writes]
    assert all(path.is_absolute() for path in written)
    assert written[0] == (tmp_path / "a.py").resolve()
    assert written[1] == (tmp_path / "sub" / "b.py").resolve()

    assert "Wrote 2 file(s): a.py, sub/b.py." in summary
    # The summary is workspace-relative, so the absolute workspace never leaks into it.
    assert str(tmp_path) not in summary


async def test_run_without_blocks_writes_nothing(tmp_path):
    agent, client = _worker(tmp_path, "I am afraid I cannot help with that today.")

    summary = await agent.run()

    assert client.writes == []
    assert list(tmp_path.iterdir()) == []
    assert "No file blocks in response, so nothing was written." in summary
    assert "I am afraid I cannot help with that today." in summary


async def test_run_rejects_an_escaping_path_and_still_writes_the_safe_one(tmp_path):
    response = "FILE: ../evil.py\n```\npwned\n```\nFILE: safe.py\n```\nok\n```\n"
    agent, client = _worker(tmp_path, response)

    summary = await agent.run()

    assert not (tmp_path.parent / "evil.py").exists()
    assert [Path(path) for path, _ in client.writes] == [(tmp_path / "safe.py").resolve()]
    assert (tmp_path / "safe.py").read_text() == "ok\n"
    assert "Wrote 1 file(s): safe.py." in summary
    assert "Rejected 1 unsafe path(s): ../evil.py." in summary


async def test_run_rejects_an_absolute_path(tmp_path):
    response = "FILE: /tmp/evil.py\n```\npwned\n```\n"
    agent, client = _worker(tmp_path, response)

    summary = await agent.run()

    assert client.writes == []
    assert not Path("/tmp/evil.py").exists()
    assert "Rejected 1 unsafe path(s): /tmp/evil.py." in summary
    assert "nothing was written" in summary


async def test_run_passes_the_task_description_to_the_model(tmp_path):
    settings = Settings(_env_file=None, workspace=tmp_path)
    client = _FakeClient("no blocks here", settings)
    agent = WorkerAgent(
        TaskModel(id="1", description="describe the widget"),
        client=client,
        settings=settings,
    )

    await agent.run()

    prompt, model = client.prompts[0]
    assert "describe the widget" in prompt
    assert model == settings.worker_model
