"""Path-safety helpers so model-generated file paths cannot escape the project.

Everything the planner/worker agents hand back is attacker-controlled as far as
the filesystem is concerned (it is LLM output), so every read/write goes
through :meth:`Workspace.resolve` first.  The rules are deliberately strict:
when in doubt it raises :class:`WorkspaceError` rather than touching a file
outside the project.

Stdlib only -- no imports from ``furrow.agents`` (and none from anywhere else in
the package), so it is safe to import from any layer.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

__all__ = ["PRUNED_DIRS", "PRUNED_SUFFIXES", "Workspace", "WorkspaceError"]

#: Directories never walked by :meth:`Workspace.iter_files` (case-insensitive).
PRUNED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".mypy_cache",
        ".ruff_cache",
        ".furrow",
        ".tox",
    }
)

#: Directory-name suffixes that are pruned as well (e.g. ``furrow.egg-info``).
PRUNED_SUFFIXES: tuple[str, ...] = (".egg-info",)


class WorkspaceError(ValueError):
    """Raised when a requested path is unsafe, or escapes the workspace root."""


def _within(root: Path, candidate: Path) -> bool:
    """Return ``True`` when ``candidate`` is ``root`` or lives underneath it.

    Uses whole path components (never a string prefix), so ``/srv/app-data``
    is *not* considered to be inside ``/srv/app``.
    """
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _is_pruned(name: str) -> bool:
    """Return ``True`` for directories excluded from traversal."""
    lowered = name.lower()
    return lowered in PRUNED_DIRS or any(lowered.endswith(s) for s in PRUNED_SUFFIXES)


@dataclass(frozen=True, slots=True)
class Workspace:
    """A project root plus guarded helpers for reading and writing files.

    The workspace is created as ``Workspace(root=path)``; ``root`` is expanded
    and resolved once at construction time so every later check compares real
    paths.  The class is frozen and hashable.
    """

    root: Path

    def __post_init__(self) -> None:
        # frozen dataclass -> bypass the setattr guard for the normalisation.
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @classmethod
    def from_settings(cls, settings: Any) -> Workspace:
        """Build a workspace from a settings object.

        Defensive: ``settings.workspace`` may not exist yet (or may be ``None``),
        in which case the current directory is used.  The path is resolved so
        containment checks compare apples to apples.
        """
        raw = getattr(settings, "workspace", None) or "."
        return cls(Path(str(raw)).expanduser())

    def resolve(self, relative: str | Path) -> Path:
        """Resolve ``relative`` inside the workspace or raise :class:`WorkspaceError`.

        Rejected inputs, in the order they are checked:

        1. **Empty / blank** paths (``""``).
        2. **NUL bytes**, which truncate the path at the syscall boundary.
        3. **Absolute paths** -- POSIX (``/etc/passwd``, ``//host/share``) and
           Windows-style (``C:\\x``, ``\\\\server\\share``).
        4. **Traversal** -- any path that climbs out of the root with ``..``
           (``../x``, ``a/../../x``).  ``a/../b`` is *not* traversal.
        5. **Symlink escapes** -- when the target does not exist yet, its
           existing ancestor is resolved with :func:`os.path.realpath` and must
           still be inside the real root, so ``link_to_etc/passwd`` is refused
           while ``link_to_cache/file.json`` is fine.
        6. **Non-directory parents** -- ``file.txt/child`` where ``file.txt``
           exists as a file.

        The returned path is absolute and normalised but is *not* required to
        exist.
        """
        raw = str(relative)
        if not raw.strip():
            raise WorkspaceError("path must not be empty")
        if "\x00" in raw:
            raise WorkspaceError(f"path must not contain NUL bytes: {raw!r}")
        if os.path.isabs(raw) or PureWindowsPath(raw).anchor:
            raise WorkspaceError(f"absolute paths are not allowed: {raw!r}")

        candidate = Path(os.path.normpath(self.root / raw))
        if not _within(self.root, candidate):
            raise WorkspaceError(f"path escapes the workspace root: {raw!r}")
        real = Path(os.path.realpath(candidate))
        if not _within(Path(os.path.realpath(self.root)), real):
            raise WorkspaceError(f"path resolves outside the workspace root (symlink): {raw!r}")

        parent = candidate.parent
        try:
            if parent.exists() and not parent.is_dir():
                raise WorkspaceError(f"parent is not a directory: {parent}")
        except OSError:
            pass
        return candidate

    def is_contained(self, candidate: Path) -> bool:
        """Return ``True`` when ``candidate`` really lives inside the root.

        Both the lexical path and its ``realpath`` are checked, so symlinks that
        point outside the workspace are reported as *not* contained.  Never
        raises -- unusable input is simply "not contained".
        """
        try:
            path = Path(candidate)
            if not path.is_absolute():
                path = self.root / path
            lexical = Path(os.path.normpath(path))
            if not _within(self.root, lexical):
                return False
            return _within(Path(os.path.realpath(self.root)), Path(os.path.realpath(lexical)))
        except (OSError, TypeError, ValueError):
            return False

    def safe_write(self, relative: str | Path, content: str) -> Path:
        """Write ``content`` as UTF-8 to ``relative``, creating parents.

        Returns the absolute path written.  Raises :class:`WorkspaceError` for
        unsafe paths or when the target is an existing directory.
        """
        target = self.resolve(relative)
        if target.is_dir():
            raise WorkspaceError(f"cannot write to a directory: {target}")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise WorkspaceError(f"could not write {target}: {exc}") from exc
        return target

    def read_text(self, relative: str | Path) -> str:
        """Read ``relative`` as UTF-8 text after a containment check."""
        return self.resolve(relative).read_text(encoding="utf-8")

    def exists(self, relative: str | Path) -> bool:
        """Return ``True`` if ``relative`` is a safe path that exists."""
        try:
            return self.resolve(relative).exists()
        except WorkspaceError:
            return False

    def _iter_paths(self) -> Iterator[Path]:
        """Yield absolute paths of contained files, pruning noisy directories."""
        root = self.root
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(name for name in dirnames if not _is_pruned(name))
            for name in sorted(filenames):
                full = Path(dirpath) / name
                # os.walk does not follow symlinked directories, but it does list
                # symlinked *files*; drop the ones aimed outside the workspace.
                if full.is_symlink() and not self.is_contained(full):
                    continue
                yield full

    def iter_files(self, limit: int = 2000) -> list[str]:
        """Return relative POSIX paths of workspace files.

        Directories listed in :data:`PRUNED_DIRS` and ``*.egg-info`` folders are
        pruned, symlinks leaving the workspace are skipped, and at most
        ``limit`` entries are returned (``0`` or less returns an empty list).
        """
        if limit <= 0:
            return []
        out: list[str] = []
        for full in self._iter_paths():
            if len(out) >= limit:
                break
            try:
                rel = full.relative_to(self.root)
            except ValueError:  # pragma: no cover - defensive
                continue
            out.append(rel.as_posix())
        return out

    def snapshot(self, limit: int = 2000) -> dict[str, float]:
        """Return ``{relative posix path: mtime}`` for change detection.

        The orchestrator diffs two consecutive snapshots to detect a stagnant
        cycle.  It is cheap -- one ``lstat`` per file, no content reads -- and
        never follows symlinks, so a link out of the workspace can neither be
        followed nor make a file look unchanged.
        """
        out: dict[str, float] = {}
        for rel in self.iter_files(limit=limit):
            try:
                out[rel] = os.lstat(self.root / rel).st_mtime
            except OSError:  # file vanished mid-walk
                continue
        return out
