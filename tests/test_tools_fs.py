"""Workspace-confined filesystem access.

Agents act on paths that come out of an LLM, so every entry point here has to reject a
path that leaves the workspace -- including one that only leaves it after symlink
resolution -- and writes have to be atomic. ``list_files`` feeds a prompt, so it is
bounded, filtered and workspace-relative.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from furrow.tools.fs import (
    DEFAULT_IGNORES,
    list_files,
    read_text,
    resolve_in_workspace,
    write_text,
)

from .conftest import run_coroutine


def make_tree(workspace: Path) -> Path:
    """Create a small tree covering kept entries, ignored entries and a hidden file."""
    (workspace / "README.md").write_text("readme\n", encoding="utf-8")
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (workspace / "src" / "app.js").write_text("console.log(1)\n", encoding="utf-8")
    (workspace / ".env").write_text("SECRET=1\n", encoding="utf-8")
    for ignored in ("__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache"):
        (workspace / ignored).mkdir()
        (workspace / ignored / "junk.pyc").write_text("junk\n", encoding="utf-8")
    (workspace / "dist").mkdir()
    (workspace / "dist" / "bundle.js").write_text("bundled\n", encoding="utf-8")
    (workspace / "furrow.egg-info").mkdir()
    (workspace / "furrow.egg-info" / "PKG-INFO").write_text("info\n", encoding="utf-8")
    return workspace


# ------------------------------------------------------------------- listing


def test_listing_is_workspace_relative_and_posix(workspace: Path) -> None:
    make_tree(workspace)
    listed = run_coroutine(list_files(workspace))
    assert "src/app.py" in listed
    assert all(not name.startswith("/") for name in listed)
    assert sorted(listed) == listed


def test_listing_skips_every_default_ignore(workspace: Path) -> None:
    make_tree(workspace)
    listed = run_coroutine(list_files(workspace))
    for ignored in DEFAULT_IGNORES:
        assert not [name for name in listed if ignored in name.split("/")]


def test_listing_skips_hidden_entries(workspace: Path) -> None:
    make_tree(workspace)
    assert ".env" not in run_coroutine(list_files(workspace))


def test_listing_skips_a_globbed_ignore(workspace: Path) -> None:
    make_tree(workspace)
    listed = run_coroutine(list_files(workspace))
    assert "furrow.egg-info/PKG-INFO" not in listed


def test_listing_honours_the_limit(workspace: Path) -> None:
    make_tree(workspace)
    assert len(run_coroutine(list_files(workspace, limit=2))) == 2


def test_listing_honours_max_depth(workspace: Path) -> None:
    (workspace / "a").mkdir()
    (workspace / "a" / "b").mkdir()
    (workspace / "a" / "b" / "deep.txt").write_text("x\n", encoding="utf-8")
    (workspace / "a" / "mid.txt").write_text("x\n", encoding="utf-8")
    (workspace / "top.txt").write_text("x\n", encoding="utf-8")
    assert run_coroutine(list_files(workspace, max_depth=1)) == ["top.txt"]
    assert run_coroutine(list_files(workspace, max_depth=2)) == ["a/mid.txt", "top.txt"]
    assert run_coroutine(list_files(workspace, max_depth=3)) == [
        "a/b/deep.txt",
        "a/mid.txt",
        "top.txt",
    ]


def test_listing_never_follows_a_symlink(workspace: Path, outside_dir: Path) -> None:
    (outside_dir / "secret.txt").write_text("secret\n", encoding="utf-8")
    (workspace / "link").symlink_to(outside_dir, target_is_directory=True)
    (workspace / "real.txt").write_text("x\n", encoding="utf-8")
    assert run_coroutine(list_files(workspace)) == ["real.txt"]


def test_listing_a_non_positive_limit_is_empty(workspace: Path) -> None:
    make_tree(workspace)
    assert run_coroutine(list_files(workspace, limit=0)) == []


def test_listing_a_missing_directory_is_empty(workspace: Path) -> None:
    assert run_coroutine(list_files(workspace / "missing")) == []


# ----------------------------------------------------------------- resolution


def test_a_relative_path_resolves_inside_the_workspace(workspace: Path) -> None:
    resolved = resolve_in_workspace("src/app.py", workspace)
    assert resolved == (workspace / "src" / "app.py").resolve()


def test_the_workspace_root_itself_resolves(workspace: Path) -> None:
    assert resolve_in_workspace(".", workspace) == workspace.resolve()


@pytest.mark.parametrize("escape", ["../outside.txt", "src/../../outside.txt", "a/b/../../../x"])
def test_a_parent_traversal_is_refused(workspace: Path, escape: str) -> None:
    with pytest.raises(ValueError, match="escapes workspace"):
        resolve_in_workspace(escape, workspace)


def test_an_absolute_path_outside_the_workspace_is_refused(
    workspace: Path, outside_dir: Path
) -> None:
    with pytest.raises(ValueError, match="escapes workspace"):
        resolve_in_workspace(outside_dir / "elsewhere.txt", workspace)


def test_a_symlink_pointing_outside_is_refused(workspace: Path, outside_dir: Path) -> None:
    target = outside_dir / "outside.txt"
    target.write_text("secret\n", encoding="utf-8")
    link = workspace / "link.txt"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="escapes workspace"):
        resolve_in_workspace("link.txt", workspace)


def test_a_traversal_that_is_cancelled_out_is_allowed(workspace: Path) -> None:
    (workspace / "src").mkdir()
    assert (
        resolve_in_workspace("src/../README.md", workspace) == (workspace / "README.md").resolve()
    )


# --------------------------------------------------------------------- reads


def test_read_text_returns_the_file_contents(workspace: Path) -> None:
    (workspace / "a.txt").write_text("hello\n", encoding="utf-8")
    assert run_coroutine(read_text("a.txt", workspace=workspace)) == "hello\n"


def test_read_text_refuses_a_path_outside_the_workspace(workspace: Path) -> None:
    with pytest.raises(ValueError, match="escapes workspace"):
        run_coroutine(read_text("../outside.txt", workspace=workspace))


def test_read_text_refuses_an_absolute_path_outside_the_workspace(
    workspace: Path, outside_dir: Path
) -> None:
    with pytest.raises(ValueError, match="escapes workspace"):
        run_coroutine(read_text(outside_dir / "elsewhere.txt", workspace=workspace))


def test_read_text_refuses_a_symlink_out_of_the_workspace(
    workspace: Path, outside_dir: Path
) -> None:
    target = outside_dir / "outside.txt"
    target.write_text("secret\n", encoding="utf-8")
    (workspace / "link.txt").symlink_to(target)
    with pytest.raises(ValueError, match="escapes workspace"):
        run_coroutine(read_text("link.txt", workspace=workspace))


def test_read_text_truncates_at_the_byte_limit(workspace: Path) -> None:
    (workspace / "big.txt").write_text("abcdef", encoding="utf-8")
    assert run_coroutine(read_text("big.txt", workspace=workspace, max_bytes=3)) == "abc"


def test_read_text_limit_counts_bytes_not_characters(workspace: Path) -> None:
    (workspace / "wide.txt").write_text("é" * 10, encoding="utf-8")
    excerpt = run_coroutine(read_text("wide.txt", workspace=workspace, max_bytes=5))
    assert excerpt == "éé\N{REPLACEMENT CHARACTER}"


def test_read_text_replaces_invalid_utf8_instead_of_raising(workspace: Path) -> None:
    (workspace / "binary.dat").write_bytes(b"\xff\xfe ok")
    excerpt = run_coroutine(read_text("binary.dat", workspace=workspace))
    assert excerpt == "\N{REPLACEMENT CHARACTER}\N{REPLACEMENT CHARACTER} ok"


# -------------------------------------------------------------------- writes


def test_write_text_creates_the_file(workspace: Path) -> None:
    written = run_coroutine(write_text("a.txt", "hello\n", workspace=workspace))
    assert written == (workspace / "a.txt").resolve()
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "hello\n"


def test_write_text_creates_parent_directories(workspace: Path) -> None:
    run_coroutine(write_text("deep/nested/a.txt", "hello", workspace=workspace))
    assert (workspace / "deep" / "nested" / "a.txt").read_text(encoding="utf-8") == "hello"


def test_write_text_replaces_existing_content(workspace: Path) -> None:
    (workspace / "a.txt").write_text("old content that is longer\n", encoding="utf-8")
    run_coroutine(write_text("a.txt", "new\n", workspace=workspace))
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "new\n"


def test_write_text_leaves_no_temporary_file_behind(workspace: Path) -> None:
    run_coroutine(write_text("a.txt", "hello", workspace=workspace))
    assert [entry.name for entry in workspace.iterdir() if entry.is_file()] == ["a.txt"]


def test_write_text_leaves_no_temporary_file_behind_on_repeat(workspace: Path) -> None:
    for index in range(3):
        run_coroutine(write_text("a.txt", f"attempt {index}", workspace=workspace))
    assert [entry.name for entry in workspace.iterdir() if entry.is_file()] == ["a.txt"]


def test_write_text_leaves_no_temporary_file_behind_on_failure(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import furrow.tools.fs as fs_module

    def exploding_replace(source: Path, target: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(fs_module.os, "replace", exploding_replace)
    with pytest.raises(OSError, match="disk full"):
        run_coroutine(write_text("a.txt", "hello", workspace=workspace))
    assert [entry.name for entry in workspace.iterdir() if entry.is_file()] == []


def test_write_text_refuses_a_path_outside_the_workspace(workspace: Path) -> None:
    with pytest.raises(ValueError, match="escapes workspace"):
        run_coroutine(write_text("../escaped.txt", "x", workspace=workspace))


def test_write_text_refuses_an_absolute_path_outside_the_workspace(
    workspace: Path, outside_dir: Path
) -> None:
    with pytest.raises(ValueError, match="escapes workspace"):
        run_coroutine(write_text(outside_dir / "escaped.txt", "x", workspace=workspace))


def test_write_text_refuses_a_symlink_out_of_the_workspace(
    workspace: Path, outside_dir: Path
) -> None:
    target = outside_dir / "outside.txt"
    target.write_text("secret\n", encoding="utf-8")
    (workspace / "link.txt").symlink_to(target)
    with pytest.raises(ValueError, match="escapes workspace"):
        run_coroutine(write_text("link.txt", "overwritten", workspace=workspace))
    assert target.read_text(encoding="utf-8") == "secret\n"


def test_write_text_round_trips_through_read_text(workspace: Path) -> None:
    content = "line one\nline two\n"
    run_coroutine(write_text("round/trip.txt", content, workspace=workspace))
    assert run_coroutine(read_text("round/trip.txt", workspace=workspace)) == content
