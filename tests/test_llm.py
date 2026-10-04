"""Offline tests for the :class:`~furrow.llm.LLMClient` file helpers.

Nothing here touches a provider: the constructor is lazy, so a client built with
throwaway settings can exercise ``read_file``, ``write_file`` and ``list_files``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from furrow.config import Settings
from furrow.llm import LLMClient, _is_ignored

CONTENT = """\
def greet(name: str) -> str:
    return f"hello {name}"
"""


def make_client(workspace: Path) -> LLMClient:
    return LLMClient(settings=Settings(_env_file=None, workspace=workspace))


async def test_write_and_read_file(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    target = tmp_path / "pkg" / "greet.py"

    # write_file is what creates the missing parent directory.
    assert target.parent.is_dir() is False

    await client.write_file(target, CONTENT)
    assert target.read_text(encoding="utf-8") == CONTENT
    assert await client.read_file(target) == CONTENT


async def test_write_file_creates_parent_directories(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    target = tmp_path / "deeply" / "nested" / "dir" / "mod.py"

    await client.write_file(target, CONTENT)

    assert await client.read_file(target) == CONTENT


async def test_write_file_overwrites_existing_content(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    target = tmp_path / "mod.py"
    await client.write_file(target, "old\n")

    await client.write_file(target, CONTENT)

    assert await client.read_file(target) == CONTENT


async def test_read_file_missing_path_raises(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    with pytest.raises(FileNotFoundError):
        await client.read_file(tmp_path / "nope.py")


async def test_read_file_accepts_str_path(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    target = tmp_path / "mod.py"
    await client.write_file(target, CONTENT)

    assert await client.read_file(str(target)) == CONTENT


def test_list_files_ignores_venv_and_pycache(tmp_path: Path) -> None:
    (tmp_path / "real.py").write_text(CONTENT, encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("docs\n", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "real.cpython-310.pyc").write_bytes(b"\x00\x01")
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    (venv / "lib").mkdir()
    (venv / "lib" / "thing.py").write_text("x = 1\n", encoding="utf-8")
    pycache_nested = tmp_path / "pkg" / "__pycache__"
    pycache_nested.mkdir(parents=True)
    (pycache_nested / "thing.pyc").write_bytes(b"\x00")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("1\n", encoding="utf-8")
    (tmp_path / "furrow.egg-info").mkdir()
    (tmp_path / "furrow.egg-info" / "PKG-INFO").write_text("Name: furrow\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.js").write_text("2\n", encoding="utf-8")

    files = make_client(tmp_path).list_files(tmp_path)

    assert "real.py" in files
    assert "docs/guide.md" in files
    assert not any(".venv" in name for name in files)
    assert not any("__pycache__" in name for name in files)
    assert not any("node_modules" in name for name in files)
    assert not any("egg-info" in name for name in files)
    assert not any(name.startswith(".git") for name in files)
    assert not any(name.startswith("build") for name in files)
    assert sorted(files) == ["docs/guide.md", "real.py"]


def test_list_files_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert make_client(tmp_path).list_files(tmp_path / "does-not-exist") == []


def test_list_files_on_empty_directory(tmp_path: Path) -> None:
    assert make_client(tmp_path).list_files(tmp_path) == []


def test_list_files_accepts_str_path(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")

    assert make_client(tmp_path).list_files(str(tmp_path)) == ["a.py"]


@pytest.mark.parametrize(
    ("path", "ignored"),
    [
        (".venv/lib/x.py", True),
        ("venv/lib/x.py", True),
        ("env/x.py", True),
        ("__pycache__/x.pyc", True),
        ("pkg/__pycache__/x.pyc", True),
        (".git/config", True),
        ("node_modules/dep/index.js", True),
        ("furrow.egg-info/PKG-INFO", True),
        ("dist/x.whl", True),
        ("build/x.o", True),
        (".tox/py/x.py", True),
        ("furrow/agents/worker.py", False),
        ("tests/test_core.py", False),
    ],
)
def test_is_ignored_matches_any_path_part(path: str, ignored: bool) -> None:
    assert _is_ignored(Path(path)) is ignored
