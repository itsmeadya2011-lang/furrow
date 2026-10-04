"""Sandboxed tool layer for Furrow worker agents.

Every capability a worker has to reach the outside world - reading and writing files,
listing the tree, searching text, running shell commands - lives here and goes through
:class:`ToolRegistry`. Two properties are load bearing:

* **Sandbox.** Every path is resolved against ``ToolContext.workspace`` and refused when
  it escapes, including via ``..``, an absolute path or a symlink pointing outside. Only
  resolved, symlink-free paths are ever opened.
* **No surprises.** :meth:`ToolRegistry.invoke` never raises for a bad tool name, bad
  arguments or a failing tool; it returns a readable string the agent loop hands back to
  the model so the model can recover on its next turn.

Arguments arrive as untrusted JSON decoded from a language model, so they are coerced
defensively instead of trusted.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import inspect
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.config import Settings
from furrow.events import EventBus, EventType

if TYPE_CHECKING:
    from furrow.llm import ToolSpec

MAX_READ_BYTES = 200_000
MAX_OUTPUT_CHARS = 20_000
MAX_LIST_RESULTS = 500
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_LINE_CHARS = 200
MAX_COMMAND_TIMEOUT = 600.0
SNIPPET_CHARS = 60
# SECURITY: hard ceiling on the WHOLE assembled `tool(args)` event message, tool name
# included - not merely on the argument list. Clipping the argument list and then wrapping
# it as f"{name}({preview})" left the true worst case at PREVIEW_CHARS + len(name) + 2, so
# a reader of the constant could not trust it as the message bound. `_preview` clips the
# assembled string instead, so `len(event.message) <= PREVIEW_CHARS` always.
PREVIEW_CHARS = 240

# SECURITY: the ONLY argument names whose value may be rendered into a `tool_call` event
# message. Every entry is a scalar the model names rather than copies: a workspace-relative
# path, a glob, a flag or a number.
#
# Everything else is DROPPED, never truncated-and-shown. The content-bearing arguments are
# `write_file.content` and `edit_file.old_string` / `edit_file.new_string`: a model
# routinely pastes real file text (and sometimes a credential it just read) into them, and
# the event message reaches the EventBus ring buffer, the `/ws` live stream and
# `GET /api/runs/{id}`. Rendering them leaked file contents off the workspace into a feed
# any event reader can read, contradicting the guarantee in README.md and .kilo/FURROW.md.
# Adding an argument here is a deliberate act: only do it if the value can never be text
# the workspace holds.
#
# `search.query` is NOT on this list and must not be added back. `search` is a LITERAL
# SUBSTRING search, so its `query` is by construction text lifted verbatim out of a file the
# workspace holds - searching for `password = LEAKCANARY_...` puts that exact secret into
# the argument. An earlier version of this comment called it "an identifier-like search
# term supplied by the caller, not file content"; that was wrong, and an adversarial pass
# reproduced the leak end to end (marker present in `GET /api/runs/{id}`, `/events` and a
# `/ws` frame). The event `data` still carries `{path, matches, tool, ok, ms}` - the match
# COUNT is what matters operationally - so `search(path=.)` stays readable without the text.
#
# `run_command.command` IS deliberately still allowed. It is the single most useful live
# observability signal, it is clipped to SNIPPET_CHARS head/tail, and it is echoed in
# `event.data["command"]` by the tool's own detail payload regardless - so removing it from
# the message would not close the route, only make it harder to see. The consequence is
# documented, not hidden: a caller must never embed a secret in a shell command.
PREVIEW_SAFE_ARGS: tuple[str, ...] = (
    "path",
    "command",
    "pattern",
    "replace_all",
    "max_bytes",
    "max_results",
    "timeout",
    "file",
    "dir",
    "glob",
)

# A tool returns its textual result plus a small, JSON-serialisable event payload.
ToolOutcome = tuple[str, dict[str, Any]]
ToolFn = Callable[..., Any]


class ToolError(Exception):
    """A recoverable tool failure, rendered as text for the model instead of raised."""


@dataclass(slots=True)
class ToolContext:
    """Everything a registry needs: the sandbox root, the policy and the activity bus."""

    workspace: Path
    settings: Settings
    bus: EventBus | None = None


def _clip(text: str, limit: int, unit: str = "characters") -> str:
    """Cap `text` at `limit` chars, keeping the head and tail around a marker.

    FIX (test suite): the tail slice was indexed from the FRONT of the string
    (`text[keep - head :]`), so a clipped 120k-char read came back at ~127k chars - the
    documented `limit` was never enforced and short inputs could be made longer. The
    result is now exactly `limit` chars whenever the input is longer.
    """
    if len(text) <= limit:
        return text
    marker = f"\n... [{len(text) - limit} {unit} truncated] ...\n"
    keep = max(0, limit - len(marker))
    head = keep * 2 // 3
    return text[:head] + marker + text[len(text) - (keep - head) :]


def _str_arg(
    arguments: dict[str, Any], key: str, *, required: bool = True, default: str = ""
) -> str:
    value = arguments.get(key)
    if value is None:
        if required:
            raise ToolError(f"missing required argument {key!r}")
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    raise ToolError(f"argument {key!r} must be a string, got {type(value).__name__}")


def _num_arg(
    arguments: dict[str, Any],
    key: str,
    default: float,
    *,
    cast: Callable[[Any], Any] = int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> int | float:
    """Coerce an untrusted number-ish argument and clamp it into range."""
    value = arguments.get(key)
    if value is None or value == "":
        parsed = cast(default)
    elif isinstance(value, bool):
        parsed = cast(value)
    elif isinstance(value, (int, float, str)):
        try:
            parsed = cast(value.strip() if isinstance(value, str) else value)
        except (TypeError, ValueError) as exc:
            raise ToolError(f"argument {key!r} must be a number, got {value!r}") from exc
    else:
        raise ToolError(f"argument {key!r} must be a number, got {type(value).__name__}")
    if minimum is not None:
        parsed = max(parsed, minimum)
    if maximum is not None:
        parsed = min(parsed, maximum)
    return parsed


def _bool_arg(arguments: dict[str, Any], key: str, default: bool = False) -> bool:
    value = arguments.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1", "on"}:
            return True
        if text in {"false", "no", "0", "off"}:
            return False
    raise ToolError(f"argument {key!r} must be a boolean, got {value!r}")


def _preview_scalar(value: Any) -> str | None:
    """Render an allowlisted argument value, or ``None`` when it is not a safe scalar.

    Arguments are untrusted JSON, so even an allowlisted name can arrive wrapped in a dict
    or a list (`{"path": {"text": "sk-..."}}`). ``repr`` of such a value would smuggle
    content straight back into the message, so non-scalars are skipped entirely rather than
    stringified.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _clip(value, SNIPPET_CHARS)
    if isinstance(value, int | float):
        return str(value)
    return None


def _preview(name: str, arguments: dict[str, Any]) -> str:
    """Assemble the `tool(args)` event message for a call. Never file contents.

    Allowlist, not a positional window: see :data:`PREVIEW_SAFE_ARGS`. Keys are walked in
    the allowlist's own order so the message does not depend on JSON key order, and only
    scalars are rendered. A content-bearing argument contributes nothing at all - there is
    deliberately no placeholder, because a placeholder sized like the value could itself
    carry content.

    The cap is applied to the ASSEMBLED message, tool name included, so the invariant a
    reader of :data:`PREVIEW_CHARS` can rely on is exactly ``len(message) <= PREVIEW_CHARS``.
    """
    parts: list[str] = []
    for key in PREVIEW_SAFE_ARGS:
        if key not in arguments:
            continue
        rendered = _preview_scalar(arguments[key])
        if rendered is not None:
            parts.append(f"{key}={rendered}")
    return _clip(f"{name}({', '.join(parts)})", PREVIEW_CHARS)


class ToolRegistry:
    """Provider-neutral tool definitions plus the sandbox that executes them."""

    def __init__(self, context: ToolContext) -> None:
        self._context = context
        try:
            self._root = Path(context.workspace).resolve()
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f"workspace {context.workspace!s} is unusable: {exc}") from exc
        if not self._root.is_dir():
            raise RuntimeError(f"workspace {self._root} is not an existing directory")
        self._excludes = tuple(context.settings.exclude_dirs or ())
        self._changed: list[str] = []
        self._specs = _build_specs()
        self._tools: dict[str, ToolFn] = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "list_files": self._list_files,
            "search": self._search,
            "run_command": self._run_command,
        }

    @property
    def workspace(self) -> Path:
        """The sandbox root every path is resolved against."""
        return self._root

    @property
    def changed_files(self) -> list[str]:
        """Workspace-relative paths this registry has written or edited, in order."""
        return list(self._changed)

    def specs(self) -> list[ToolSpec]:
        """Return JSON tool definitions. A fresh deep copy, safe to hand to a provider."""
        return json.loads(json.dumps(self._specs))

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one tool and return its textual result.

        Never raises for an unknown tool, bad arguments or a failing tool: those come back
        as an ``ERROR: ...`` string the agent loop can feed to the model for recovery.
        """
        args: dict[str, Any] = dict(arguments) if isinstance(arguments, dict) else {}
        started = time.monotonic()
        detail: dict[str, Any] = {}
        try:
            tool = self._tools.get(name)
            if tool is None:
                raise ToolError(
                    f"unknown tool {name!r}; available tools: {', '.join(sorted(self._tools))}"
                )
            if inspect.iscoroutinefunction(tool):
                text, detail = await tool(args)
            else:
                # File IO in a thread: several workers share one event loop.
                text, detail = await asyncio.to_thread(tool, args)
            ok = True
        except ToolError as exc:
            ok, text, detail = False, f"ERROR: {exc}", {}
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # defensive: a broken tool must not kill the agent
            ok, text, detail = False, f"ERROR: {name} failed unexpectedly: {exc!r}", {}
        self._emit(name, ok, args, {**detail, "tool": name, "ok": ok}, started)
        return text

    # -------------------------------------------------------------- internals

    def _emit(
        self, name: str, ok: bool, arguments: dict[str, Any], detail: dict[str, Any], start: float
    ) -> None:
        bus = self._context.bus
        if bus is None:
            return
        payload: dict[str, Any] = {**detail, "ms": int((time.monotonic() - start) * 1000)}
        try:
            json.dumps(payload)
        except (TypeError, ValueError):
            payload = {"tool": name, "ok": ok}
        # Security: only the allowlisted identifier arguments reach the message; content-
        # bearing arguments are dropped. See PREVIEW_SAFE_ARGS. The cap covers the whole
        # message, so no single tool call can blow up the ring buffer.
        bus.emit(EventType.TOOL_CALL, _preview(name, arguments), **payload)

    def _resolve(self, path: str, *, label: str = "path") -> Path:
        """Resolve `path` inside the sandbox or refuse, naming the offending path."""
        raw = (path or "").strip()
        candidate = Path(raw) if raw else Path(".")
        target = candidate if candidate.is_absolute() else self._root / candidate
        try:
            resolved = target.resolve()
        except (OSError, RuntimeError) as exc:
            raise ToolError(f"cannot resolve {label} {raw!r}: {exc}") from exc
        if not resolved.is_relative_to(self._root):
            raise ToolError(
                f"refusing to use {label} {raw!r}: it resolves to {resolved}, which is "
                f"outside the workspace sandbox {self._root}"
            )
        return resolved

    def _rel(self, target: Path) -> str:
        try:
            return target.relative_to(self._root).as_posix() or "."
        except ValueError:
            return target.name

    def _excluded(self, name: str) -> bool:
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in self._excludes)

    def _walk(self, root: Path) -> Iterator[tuple[Path, str]]:
        """Yield `(absolute, workspace-relative)` for real, in-sandbox files."""
        stack = [root]
        while stack:
            try:
                entries = sorted(stack.pop().iterdir(), key=lambda item: item.name)
            except (OSError, ValueError):
                continue
            subdirs: list[Path] = []
            for entry in entries:
                if self._excluded(entry.name):
                    continue
                try:
                    if not entry.resolve().is_relative_to(self._root):
                        continue  # symlink pointing out of the sandbox
                    if entry.is_dir():
                        subdirs.append(entry)
                    elif entry.is_file():
                        yield entry, self._rel(entry)
                except (OSError, RuntimeError):
                    continue
            stack.extend(reversed(subdirs))

    def _atomic_write(self, target: Path, blob: bytes) -> None:
        """Write via a temp file in the same directory, then `os.replace` over the target."""
        tmp_name = ""
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".furrow-tmp")
            with os.fdopen(handle, "wb") as stream:
                stream.write(blob)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_name, target)
        except OSError as exc:
            if tmp_name:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name)
            raise ToolError(f"could not write {self._rel(target)}: {exc}") from exc

    def _record(self, target: Path) -> str:
        rel = self._rel(target)
        if rel not in self._changed:
            self._changed.append(rel)
        return rel

    def _require_file(self, target: Path) -> None:
        if not target.exists():
            raise ToolError(f"{self._rel(target)} does not exist; read or list it first")
        if target.is_dir():
            raise ToolError(f"{self._rel(target)} is a directory, not a file; use list_files")

    # ------------------------------------------------------------------ tools

    def _read_file(self, arguments: dict[str, Any]) -> ToolOutcome:
        target = self._resolve(_str_arg(arguments, "path"))
        max_bytes = int(_num_arg(arguments, "max_bytes", MAX_READ_BYTES, minimum=1))
        self._require_file(target)
        size = target.stat().st_size
        with target.open("rb") as stream:
            blob = stream.read(max_bytes)
        if b"\x00" in blob:
            raise ToolError(
                f"{self._rel(target)} looks binary (NUL byte found); refusing to read it as text"
            )
        rel = self._rel(target)
        text = blob.decode("utf-8", errors="replace")
        if size > len(blob):
            text += f"\n... [truncated at {max_bytes} of {size} bytes]"
        return _clip(text, MAX_OUTPUT_CHARS), {"path": rel, "bytes": len(blob)}

    def _write_file(self, arguments: dict[str, Any]) -> ToolOutcome:
        target = self._resolve(_str_arg(arguments, "path"))
        blob = _str_arg(arguments, "content").encode("utf-8")
        self._atomic_write(target, blob)
        rel = self._record(target)
        return f"Wrote {len(blob)} bytes to {rel}.", {"path": rel, "bytes": len(blob)}

    def _edit_file(self, arguments: dict[str, Any]) -> ToolOutcome:
        target = self._resolve(_str_arg(arguments, "path"))
        old = _str_arg(arguments, "old_string")
        new = _str_arg(arguments, "new_string")
        replace_all = _bool_arg(arguments, "replace_all", False)
        if not old:
            raise ToolError("old_string must not be empty; it is the text being replaced")
        self._require_file(target)
        text = target.read_text(encoding="utf-8", errors="replace")
        count, rel = text.count(old), self._rel(target)
        if count == 0:
            raise ToolError(
                f"old_string was not found in {rel}. Re-read the file and copy the text exactly, "
                "including indentation and trailing newlines."
            )
        if count > 1 and not replace_all:
            raise ToolError(
                f"old_string is ambiguous: it appears {count} times in {rel}. Add more "
                "surrounding lines to make it unique, or set replace_all=true to replace "
                "all of them."
            )
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        self._atomic_write(target, updated.encode("utf-8"))
        self._record(target)
        return (
            f"Made {count} replacement(s) in {rel}.",
            {"path": rel, "replacements": count},
        )

    def _list_files(self, arguments: dict[str, Any]) -> ToolOutcome:
        root = self._resolve(_str_arg(arguments, "path", required=False, default="."))
        pattern = _str_arg(arguments, "pattern", required=False, default="*") or "*"
        if not root.is_dir():
            raise ToolError(f"{self._rel(root)} is not an existing directory")
        matches: list[str] = []
        total = 0
        for _, rel in self._walk(root):
            if fnmatch.fnmatchcase(rel, pattern):
                total += 1
                if len(matches) < MAX_LIST_RESULTS:
                    matches.append(rel)
        rel_root = self._rel(root)
        if total == 0:
            return f"No files under {rel_root} match {pattern!r}.", {"path": rel_root, "files": 0}
        if total > len(matches):
            matches.append(
                f"... [{total} files match; truncated to the first {MAX_LIST_RESULTS}; paths are "
                "relative to the workspace root]"
            )
        return "\n".join(matches), {"path": rel_root, "files": total}

    def _search(self, arguments: dict[str, Any]) -> ToolOutcome:
        query = _str_arg(arguments, "query")
        if not query:
            raise ToolError("query must not be empty")
        root = self._resolve(_str_arg(arguments, "path", required=False, default="."))
        limit = int(_num_arg(arguments, "max_results", 100, minimum=1, maximum=MAX_LIST_RESULTS))
        if not root.is_dir():
            raise ToolError(f"{self._rel(root)} is not an existing directory")
        lines: list[str] = []
        truncated = False
        for absolute, rel in self._walk(root):
            try:
                if absolute.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                blob = absolute.read_bytes()
            except OSError:
                continue
            if b"\x00" in blob:
                continue
            for number, line in enumerate(blob.decode("utf-8", errors="replace").splitlines(), 1):
                if query in line:
                    lines.append(f"{rel}:{number}: {_clip(line.strip(), MAX_LINE_CHARS, 'chars')}")
                    if len(lines) > limit:
                        lines.pop()
                        truncated = True
                        break
            if truncated:
                break
        rel_root = self._rel(root)
        if not lines:
            return f"No matches for {query!r} under {rel_root}.", {"path": rel_root, "matches": 0}
        if truncated:
            lines.append(f"... [stopped after {limit} matches; narrow the query or the path]")
        return (
            _clip("\n".join(lines), MAX_OUTPUT_CHARS),
            {"path": rel_root, "matches": len(lines)},
        )

    async def _run_command(self, arguments: dict[str, Any]) -> ToolOutcome:
        command = _str_arg(arguments, "command")
        if not command.strip():
            raise ToolError("command must not be empty")
        timeout = float(
            _num_arg(
                arguments,
                "timeout",
                self._context.settings.request_timeout,
                cast=float,
                minimum=1.0,
                maximum=MAX_COMMAND_TIMEOUT,
            )
        )
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self._root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)
            except Exception:
                stdout, stderr = b"", b""
        report = f"$ {command}\n"
        report += (
            f"TIMED OUT after {timeout:.0f}s and was killed (no exit code)."
            if timed_out
            else f"exit code: {process.returncode}"
        )
        out_text = _clip(stdout.decode("utf-8", errors="replace"), MAX_OUTPUT_CHARS, "stdout chars")
        err_text = _clip(stderr.decode("utf-8", errors="replace"), MAX_OUTPUT_CHARS, "stderr chars")
        if out_text.strip():
            report += f"\n--- stdout ---\n{out_text}"
        if err_text.strip():
            report += f"\n--- stderr ---\n{err_text}"
        return report, {
            "command": _clip(command, SNIPPET_CHARS),
            "exit_code": None if timed_out else process.returncode,
            "timed_out": timed_out,
        }


def _prop(kind: str, description: str, **extra: Any) -> dict[str, Any]:
    """Build one JSON Schema property."""
    return {"type": kind, "description": description, **extra}


def _spec(name: str, summary: str, properties: dict[str, Any], required: list[str]) -> ToolSpec:
    """Build one provider-neutral tool definition."""
    return {
        "name": name,
        "description": summary,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


PATH_ARG = _prop("string", "Workspace-relative file path, e.g. 'furrow/agents/worker.py'.")
DIR_ARG = _prop("string", "Workspace-relative directory to start from. Default '.'.")


def _build_specs() -> list[ToolSpec]:
    """JSON tool definitions. The model reads these to decide which tool to call."""
    return [
        _spec(
            "read_file",
            "Read a UTF-8 text file and return its contents. Always read a file before "
            "changing it. Binary files are refused.",
            {
                "path": PATH_ARG,
                "max_bytes": _prop("integer", "Stop after this many bytes. Default 200000.",
                                   minimum=1),
            },
            ["path"],
        ),
        _spec(
            "write_file",
            "Create or overwrite a file atomically with the given content, creating parent "
            "directories as needed. Prefer edit_file for files that already exist.",
            {
                "path": PATH_ARG,
                "content": _prop("string", "Full content of the file. May be empty."),
            },
            ["path", "content"],
        ),
        _spec(
            "edit_file",
            "Replace an exact string in an existing file and report how many replacements "
            "were made. Fails if old_string is absent, and fails if it is ambiguous "
            "(appears more than once) unless replace_all is true.",
            {
                "path": PATH_ARG,
                "old_string": _prop(
                    "string",
                    "Exact existing text to replace, copied verbatim including indentation. "
                    "Add surrounding lines to make it unique.",
                ),
                "new_string": _prop("string", "Replacement text. Empty string deletes."),
                "replace_all": _prop(
                    "boolean", "Replace every occurrence instead of requiring a unique one."
                ),
            },
            ["path", "old_string", "new_string"],
        ),
        _spec(
            "list_files",
            "List files under a directory as workspace-relative paths, one per line. "
            "Excluded directories such as caches and virtualenvs are skipped.",
            {
                "path": DIR_ARG,
                "pattern": _prop(
                    "string",
                    "Glob matched against each workspace-relative path, e.g. '*.py' or "
                    "'furrow/agents/*'. Default '*'.",
                ),
            },
            [],
        ),
        _spec(
            "search",
            "Literal substring search over text files under a directory, returning "
            "'path:line: text' lines. Use it to find the code you need to change.",
            {
                "query": _prop("string", "Literal text to look for. Case sensitive, not a "
                               "regex."),
                "path": DIR_ARG,
                "max_results": _prop("integer", "Maximum matches to return. Default 100.",
                                     minimum=1),
            },
            ["query"],
        ),
        _spec(
            "run_command",
            "Run a shell command with the workspace root as the working directory and "
            "return its output plus the exit code. Use it for tests, lint and typecheck.",
            {
                "command": _prop("string", "Shell command, e.g. 'python3 -m pytest -q'."),
                "timeout": _prop(
                    "number",
                    "Seconds before the command is killed. Defaults to the configured "
                    "request timeout, capped at 600.",
                    minimum=1,
                ),
            },
            ["command"],
        ),
    ]

