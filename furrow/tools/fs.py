"""Sandboxed filesystem access for agents.

Everything here is confined to the configured workspace. Agents act on paths
that come out of an LLM, so an unvalidated ``open()`` is an arbitrary-file-read
and arbitrary-file-write primitive; :func:`resolve_in_workspace` is the single
choke point that makes a path unusable before any I/O happens.

Two more properties matter once the path is validated:

* Nothing blocks the event loop. Directory creation, directory walking and
  ``os.replace`` run in worker threads; file reads and writes go through
  ``aiofiles``.
* Writes are atomic. Content lands in a temporary sibling and is moved into place
  with :func:`os.replace`, so an interrupted write cannot leave a half-written
  source file behind -- truncating in place can.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import tempfile
from pathlib import Path

import aiofiles

DEFAULT_IGNORES: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        "*.egg-info",
    }
)

DEFAULT_READ_MAX_BYTES = 200_000
DEFAULT_LIST_LIMIT = 500
DEFAULT_MAX_DEPTH = 6


def resolve_in_workspace(path: str | Path, workspace: Path) -> Path:
    """Resolve *path* against *workspace* and return an absolute, symlink-free path.

    Raises:
        ValueError: if the resolved path is *workspace* itself or lies outside it.
    """
    root = Path(workspace).expanduser().resolve()
    candidate = Path(path).expanduser()
    # Both sides are fully resolved before comparing: Path.resolve() alone happily
    # turns "<workspace>/../etc/passwd" into "/etc/passwd", and a relative path is
    # only interpreted correctly relative to the *resolved* root.
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path {os.fspath(path)!r} escapes workspace {root}")
    return resolved


async def read_text(
    path: str | Path,
    *,
    workspace: Path,
    max_bytes: int = DEFAULT_READ_MAX_BYTES,
) -> str:
    """Read *path* from *workspace* as text, at most *max_bytes* bytes long.

    Invalid UTF-8 is replaced rather than raising, and content past the limit is
    dropped -- an agent gets a readable excerpt instead of a decoder traceback or a
    megabyte of log file in its context window.
    """
    target = resolve_in_workspace(path, workspace)
    async with aiofiles.open(target, "rb") as handle:
        raw = await handle.read(max_bytes)
    # Read as bytes so the cap is really a byte cap, then decode once.
    return raw.decode("utf-8", errors="replace")


async def write_text(path: str | Path, content: str, *, workspace: Path) -> Path:
    """Atomically write *content* to *path* inside *workspace*.

    Parent directories are created, the bytes go to a temporary sibling, and
    :func:`os.replace` moves them into place. Returns the resolved path written.
    """
    target = resolve_in_workspace(path, workspace)
    await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
    # Same directory as the target so os.replace stays within one filesystem and is
    # therefore atomic; mkstemp gives a name no concurrent writer will collide with.
    handle_fd, tmp_name = await asyncio.to_thread(
        tempfile.mkstemp,
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        async with aiofiles.open(handle_fd, "w", encoding="utf-8") as handle:
            await handle.write(content)
        await asyncio.to_thread(os.replace, tmp_path, target)
    except BaseException:
        # Never leave the partial temp file behind for the next listing to find.
        await asyncio.to_thread(_unlink_quietly, tmp_path)
        raise
    return target


async def list_files(
    workspace: Path,
    *,
    limit: int = DEFAULT_LIST_LIMIT,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> list[str]:
    """List files under *workspace* as sorted workspace-relative POSIX paths.

    Directories are walked at most *max_depth* levels deep, at most *limit* paths
    are returned, ignored and hidden entries are skipped, and symlinks are never
    followed. Bounded because the result is fed into a prompt: an unbounded
    ``rglob("*")`` over a checkout with a populated ``.git`` does not fit.
    """
    return await asyncio.to_thread(
        _collect_files, Path(workspace).expanduser().resolve(), limit, max_depth
    )


def _collect_files(root: Path, limit: int, max_depth: int) -> list[str]:
    if limit <= 0 or not root.is_dir():
        return []
    found: list[str] = []
    # Iterative walk with an explicit stack: bounded depth, no recursion limit, and
    # a cheap early exit once `limit` paths are collected.
    pending: list[tuple[Path, int]] = [(root, 0)]
    while pending and len(found) < limit:
        directory, depth = pending.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
        except OSError:
            continue  # unreadable directory: skip it rather than fail the listing
        for entry in entries:
            if len(found) >= limit:
                break
            if _is_ignored(entry.name) or entry.is_symlink():
                # A link can leave the workspace entirely, and following a link cycle
                # never terminates; neither is worth resolving for a path listing.
                continue
            if entry.is_dir():
                if depth + 1 < max_depth:
                    pending.append((entry, depth + 1))
            elif entry.is_file() and depth + 1 <= max_depth:
                found.append(entry.relative_to(root).as_posix())
    return sorted(found)


def _is_ignored(name: str) -> bool:
    """True for entry names that must not appear in a workspace listing."""
    if name in DEFAULT_IGNORES:
        return True
    if any(fnmatch.fnmatch(name, pattern) for pattern in DEFAULT_IGNORES if "*" in pattern):
        return True
    # Hidden entries (".env", ".idea", ".DS_Store", ...) are skipped as well: they
    # are secrets, editor state or caches, never something an agent needs, and the
    # ones that matter for performance are named explicitly in DEFAULT_IGNORES.
    return name.startswith(".")


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
