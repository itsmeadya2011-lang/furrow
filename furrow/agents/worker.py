from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)

# --- Context budget -------------------------------------------------------------------
# The whole point of this agent is to show the model real repository content instead of
# asking it to guess, so the caps below exist only to keep the request inside a normal
# context window: a per-file cap stops one huge module from starving the rest, a per-task
# file count stops a sprawling plan from filling the prompt with paths, and a total cap
# is the hard ceiling regardless of how the pieces add up.
MAX_FILE_CHARS = 8_000
MAX_PROMPT_CHARS = 40_000
MAX_CONTEXT_FILES = 12
MAX_DESCRIPTION_CHARS = 2_000
MAX_PATH_CHARS = 200
MAX_DEPENDENCY_IDS = 20
MAX_DEPENDENCY_RESULTS = 8
MAX_DEPENDENCY_RESULT_CHARS = 1_500
MAX_DEPENDENCY_SECTION_CHARS = 4_000
MAX_LISTING_ENTRIES = 200
MAX_SUMMARY_CHARS = 4_000
TRUNCATION_MARKER = "... [truncated]"

# --- Edit protocol ----------------------------------------------------------------------
# The model answers with one JSON envelope; everything below turns that envelope into
# files on disk. The caps here bound how much a single reply may rewrite, and the
# FORBIDDEN_* sets block the class of generated/derived trees the worker must never
# edit; they do not mirror ``LLMClient.list_files``' prune list because every dotted
# directory (including ``.furrow`` and ``.github``) is intentionally allowed here.
MAX_ISSUES = 5
MAX_ISSUE_CHARS = 300
MAX_REASON_CHARS = 300
MAX_REPORT_LINE_CHARS = 240
MAX_REPORT_SUMMARY_CHARS = 800
MAX_BACKUP_ATTEMPTS = 64
BACKUP_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
TEMP_PREFIX = ".furrow-tmp-"

ACTION_WRITE = "write"
ACTION_CREATED = "created"
ACTION_MODIFIED = "modified"
ACTION_SKIPPED = "skipped"

# Directories no worker reply may write into. ``.tox``/``.nox`` and the tool caches are
# beyond the task's list, kept because they are the same class of generated tree.
FORBIDDEN_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
        ".tox",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)
FORBIDDEN_DIR_SUFFIXES = (".egg-info",)
FORBIDDEN_FILE_SUFFIXES = (".pyc", ".pyo", ".pyd")

SECTION_TASK = "## Task"
SECTION_FILES = "Files to touch"
SECTION_DEPS = "## Completed dependencies"
SECTION_CONTEXT = "## Repository context (current file contents)"
SECTION_LISTING = "## Workspace files (no specific files were given)"

_FENCE_BY_SUFFIX = {
    ".py": "python",
    ".pyi": "python",
    ".md": "markdown",
    ".toml": "toml",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".ini": "ini",
    ".cfg": "ini",
    ".sh": "bash",
    ".sql": "sql",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".html": "html",
    ".css": "css",
    ".txt": "text",
}


class WorkerError(RuntimeError):
    """Raised when the worker cannot obtain a usable result from the LLM.

    The orchestrator collects task exceptions and stores ``str(exc)`` as the task result,
    so this message is the only thing the next planning cycle gets to work with; it names
    the task, the model and the underlying cause instead of leaking an SDK traceback.
    """


# --- Bounded-text helpers -------------------------------------------------------------


def clamp_head(text: str, limit: int) -> str:
    """Keep the first ``limit`` characters, marking that the tail was dropped.

    Used for text where the opening carries the meaning (task descriptions, dependency
    results): the reader loses detail, not the intent.
    """
    if len(text) <= limit:
        return text
    if limit <= len(TRUNCATION_MARKER):
        return TRUNCATION_MARKER[:limit]
    keep = limit - len(TRUNCATION_MARKER) - 1
    return f"{text[:keep].rstrip()}\n{TRUNCATION_MARKER}"


def clamp_middle(text: str, limit: int) -> str:
    """Keep the head and the tail of ``text`` and drop the middle.

    Source files are useful at both ends: the head carries the imports and the class and
    function signatures an edit has to match, the tail carries the last definitions and
    any entry-point code. A head-only cap would hide the bottom of a long module and a
    tail-only cap would hide its imports, so the middle is what goes.
    """
    if len(text) <= limit:
        return text
    if limit <= len(TRUNCATION_MARKER) + 2:
        return clamp_head(text, limit)
    keep = limit - len(TRUNCATION_MARKER) - 2
    head = keep - keep // 2
    tail = keep - head
    return f"{text[:head]}\n{TRUNCATION_MARKER}\n{text[len(text) - tail :]}"


def _fence_for(path: str) -> str:
    return _FENCE_BY_SUFFIX.get(Path(path).suffix.lower(), "")


def format_file_block(path: str, content: str, max_chars: int = MAX_FILE_CHARS) -> str:
    """Render one file as a fenced, size-capped block for the prompt."""
    return (
        f"### {clamp_head(path, MAX_PATH_CHARS)}\n"
        f"```{_fence_for(path)}\n{clamp_middle(content, max_chars)}\n```"
    )


def format_unavailable_file(path: str, reason: str) -> str:
    """Placeholder for a file that could not be read.

    A stale path in the plan must cost context, not the whole task, and the model has to
    be told the file is missing rather than left to infer it from an absent section.
    """
    return (
        f"### {clamp_head(path, MAX_PATH_CHARS)}\n"
        f"[unavailable: could not read this file - {clamp_head(reason, MAX_PATH_CHARS)}]"
    )


def _format_path_list(paths: Sequence[str], max_entries: int = MAX_CONTEXT_FILES) -> str:
    """Bounded target-path list so a pathological plan cannot inflate the header."""
    shown = ", ".join(clamp_head(p, MAX_PATH_CHARS) for p in paths[:max_entries])
    if len(paths) > max_entries:
        shown += f", ... (+{len(paths) - max_entries} more)"
    return shown or "any"


def format_dependencies(
    task: TaskModel, dependency_results: Mapping[str, Any] | None = None
) -> str:
    """Render the task's dependency ids plus, when supplied, what those tasks produced.

    Tasks may run in parallel and a dependency's output is the main reason the worker's
    edit is even possible, so without this the model happily re-implements or contradicts
    work that already exists. ``dependency_results`` is optional because the current
    orchestrator call site does not pass it; the ids alone still tell the worker to expect
    code it has not seen. The whole section is capped so a long dependency chain cannot
    crowd the repository context out of the prompt.
    """
    deps = list(task.dependencies)
    results = dict(dependency_results or {})
    if not deps and not results:
        return ""

    shown_ids = [clamp_head(d, MAX_PATH_CHARS) for d in deps[:MAX_DEPENDENCY_IDS]]
    lines = [f"Depends on: {', '.join(shown_ids) if shown_ids else 'unknown'}"]
    # The overflow notices precede the result bodies on purpose: the section is head-capped
    # below, so a trailing notice would be the first thing cut and the model would never
    # learn that the list it can see is incomplete.
    if len(deps) > MAX_DEPENDENCY_IDS:
        lines.append(f"- ... and {len(deps) - MAX_DEPENDENCY_IDS} further dependencies")
    if len(results) > MAX_DEPENDENCY_RESULTS:
        lines.append(f"- ... and {len(results) - MAX_DEPENDENCY_RESULTS} further results")
    for dep_id, result in list(results.items())[:MAX_DEPENDENCY_RESULTS]:
        text = clamp_head(str(result).strip(), MAX_DEPENDENCY_RESULT_CHARS)
        lines.append(f"- [task {clamp_head(dep_id, MAX_PATH_CHARS)}] {text or '(no result)'}")
    return clamp_head("\n".join(lines), MAX_DEPENDENCY_SECTION_CHARS)


def format_workspace_listing(
    paths: Sequence[str], error: str | None = None, max_entries: int = MAX_LISTING_ENTRIES
) -> str:
    """Render a bounded workspace listing, or the reason it is unavailable."""
    if error:
        return f"[workspace listing unavailable: {clamp_head(error, MAX_PATH_CHARS)}]"
    if not paths:
        return "(no files found in the workspace)"
    shown = "\n".join(clamp_head(p, MAX_PATH_CHARS) for p in paths[:max_entries])
    if len(paths) > max_entries:
        shown += f"\n... and {len(paths) - max_entries} more files"
    return shown


# --- Context gathering ----------------------------------------------------------------


def read_workspace_listing(client: LLMClient, workspace: Path) -> tuple[list[str], str | None]:
    """Best-effort workspace listing, returning ``(paths, error)``.

    A failure here costs the model some orientation, not the task, so nothing raises - except
    ``asyncio.CancelledError``, which is a ``BaseException`` and therefore propagates through
    the ``except Exception`` guards, which is what cooperative cancellation needs.

    ``LLMClient.list_files`` currently takes ``directory`` as a required positional argument,
    so the no-argument call below raises ``TypeError`` and the explicit-directory call is the
    one that does the work. Trying the no-argument form first costs one cheap failed call and
    keeps this working unchanged if that parameter later becomes optional and defaults to the
    workspace. Only the ``TypeError`` is retried; every other failure is reported as a note.
    """
    try:
        return list(client.list_files()), None
    except TypeError:
        pass
    except Exception as exc:  # noqa: BLE001 - any failure degrades to a note in the prompt
        return [], f"{type(exc).__name__}: {exc}"
    try:
        return list(client.list_files(workspace)), None
    except Exception as exc:  # noqa: BLE001 - same rationale as above
        return [], f"{type(exc).__name__}: {exc}"


async def read_source(client: LLMClient, path: str, workspace: Path) -> str:
    """Read a repo-relative path, tolerating a wrong working directory.

    ``task.files`` is repo-relative; ``LLMClient.read_file`` resolves against the process
    CWD, so a workspace-joined path is tried as well before the caller gives up.
    """
    last: Exception | None = None
    for candidate in (path, str(Path(workspace) / path)):
        try:
            return await client.read_file(candidate)
        except Exception as exc:  # noqa: BLE001 - reported to the caller as a note
            last = exc
    if last is not None:
        raise last
    raise OSError(f"no readable path for {path!r}")


async def build_context(task: TaskModel, client: LLMClient, workspace: Path) -> str:
    """Build the grounded repository-content block for ``task``.

    This is the fix for the core bug: the model previously saw a task line and a file
    list with no content at all, so it invented code. Every read is individually guarded -
    a stale or unreadable path degrades to a note, and if nothing at all could be read we
    append the workspace listing so the model still knows the repo layout.

    The dependency section is deliberately *not* built here: it is task metadata, not
    repository content, so :func:`build_prompt` renders it once in the header next to the
    task it belongs to. Rendering it in both places would spend up to
    ``MAX_DEPENDENCY_SECTION_CHARS`` twice out of the one ``MAX_PROMPT_CHARS`` budget.
    """
    if not task.files:
        paths, error = read_workspace_listing(client, workspace)
        return f"{SECTION_LISTING}\n{format_workspace_listing(paths, error)}"

    files = list(task.files[:MAX_CONTEXT_FILES])
    blocks: list[str] = []
    read_count = 0
    for rel in files:
        try:
            content = await read_source(client, rel, workspace)
        except Exception as exc:  # noqa: BLE001 - one bad path must not abort the task
            blocks.append(format_unavailable_file(rel, f"{type(exc).__name__}: {exc}"))
            continue
        read_count += 1
        blocks.append(format_file_block(rel, content))

    if len(task.files) > MAX_CONTEXT_FILES:
        blocks.append(
            f"... and {len(task.files) - MAX_CONTEXT_FILES} further target files "
            "(not shown, context cap)"
        )
    if files and read_count == 0:
        paths, error = read_workspace_listing(client, workspace)
        blocks.append(f"{SECTION_LISTING}\n{format_workspace_listing(paths, error)}")

    return f"{SECTION_CONTEXT}\n" + "\n\n".join(blocks)


def build_prompt(
    task: TaskModel,
    context: str = "",
    dependency_results: Mapping[str, Any] | None = None,
) -> str:
    """Assemble the full worker prompt. Pure and synchronous so it is unit-testable.

    The header carries the prompt text, the task, its target paths and the dependency
    section; ``context`` is the repository content built by :func:`build_context`. The two
    are joined under one hard cap so no combination of large task and large repository
    block can exceed ``MAX_PROMPT_CHARS``.
    """
    header = "\n".join(
        [
            WORKER_PROMPT.strip(),
            "",
            SECTION_TASK,
            clamp_head(task.description.strip(), MAX_DESCRIPTION_CHARS),
        ]
    )
    if task.files:
        header += f"\n\n{SECTION_FILES}: {_format_path_list(task.files)}"
    deps = format_dependencies(task, dependency_results)
    if deps:
        header += f"\n\n{SECTION_DEPS}\n{deps}"
    if not context:
        return clamp_middle(header, MAX_PROMPT_CHARS)

    budget = MAX_PROMPT_CHARS - len(header) - 2
    if budget <= 0:
        # The header alone fills the window; say so rather than let the model believe the
        # repository is really that small.
        return clamp_middle(f"{header}\n\n{TRUNCATION_MARKER}", MAX_PROMPT_CHARS)
    if len(context) > budget:
        context = clamp_middle(context, budget)
    return clamp_middle(f"{header}\n\n{context}", MAX_PROMPT_CHARS)


def format_summary(text: str) -> str:
    """Normalise the task result into the bounded blob the orchestrator stores.

    It is applied to two things: the prose reply of a run that had no edit envelope, and
    the report :func:`format_report` builds for a run that did. Both are what the next
    planning cycle reads back - what changed, which files were touched, any issues or
    blockers, and whether the task is complete. Nothing is invented here - only stripped,
    capped, and defaulted when there was nothing at all, which the orchestrator then shows
    and the goal-refinement path can act on.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return "(worker returned no output; completion unknown)"
    return clamp_head(cleaned, MAX_SUMMARY_CHARS)


# --- Envelope extraction ---------------------------------------------------------------
#
# This mirrors planner.extract_json_payload on purpose instead of importing it. The
# parser is genuinely generic, but planner and worker have different owners and
# different contracts: the planner *must* fail loudly on unusable JSON (it has a repair
# round-trip), while the worker *must not* fail - a prose reply is a valid "I changed
# nothing" answer. Sharing one parser would force the worker's tolerance onto the
# planner and its strictness onto the worker, so each module keeps its own copy of the
# string/bracket scan and only the *shape* preference differs.

_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)\n?```", re.DOTALL)

_BRACKET_PAIRS = {"{": "}", "[": "]"}
_OPENER_FOR_CLOSER = {"}": "{", "]": "["}

ENVELOPE_KEYS = frozenset({"summary", "complete", "issues", "files"})


def _strip_code_fences(text: str) -> str:
    """Return the body of the first fenced block, or the text unchanged."""
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _json_regions(text: str) -> list[str]:
    """Return every outermost balanced ``{...}``/``[...]`` span in ``text``.

    A stack scan rather than a regex, so a brace inside a JSON string does not move the
    depth and nested objects stay inside their parent. Every disjoint region is
    returned, not just the first: a chatty reply often shows a small example object
    before the real answer, and a single-region scan would stop at the decoy.
    """
    regions: list[str] = []
    stack: list[str] = []
    start = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in _BRACKET_PAIRS:
            if not stack:
                start = index
            stack.append(char)
        elif char in "}]" and stack:
            opener = _OPENER_FOR_CLOSER[char]
            while stack and stack[-1] != opener:
                stack.pop()
            if stack:
                stack.pop()
            if not stack:
                regions.append(text[start : index + 1])
    return regions


def _envelope_candidates(text: str) -> list[str]:
    """Every substring of ``text`` that could be the edit envelope, in priority order."""
    unfenced = _strip_code_fences(text)
    candidates: list[str] = []
    for candidate in (unfenced, *_json_regions(unfenced)):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _is_envelope_shaped(payload: Any) -> bool:
    """Whether a parsed payload is the worker's envelope rather than a decoy object.

    ``files`` or ``summary`` is the load-bearing signal. An object made *only* of
    envelope keys (``{"complete": true}``) also counts, so a model that omits the
    optional prose still gets its completeness flag read instead of having its raw JSON
    reported back as a summary.
    """
    if not isinstance(payload, dict):
        return False
    if "files" in payload or "summary" in payload:
        return True
    return bool(payload) and set(payload) <= ENVELOPE_KEYS


def extract_edit_envelope(text: str) -> dict[str, Any] | None:
    """Parse the worker's edit envelope, or return ``None`` for a prose reply.

    Models routinely fence the JSON in ```json or wrap it in a sentence ("Here is the
    change:"). Both are framing rather than content, so fences are dropped and every
    outermost balanced object is tried in turn, preferring the first envelope-shaped one
    so a decoy example earlier in the reply cannot win. Nothing inside a region is
    repaired: trailing commas stay invalid, because a silently reinterpreted edit list
    would write the wrong bytes to the workspace.

    ``None`` means "no envelope": the caller keeps the reply as an advisory summary and
    applies nothing. That is a supported outcome (the prompt allows a prose-only answer
    when the worker decided to change nothing), not a failure.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    for candidate in _envelope_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue  # try the next framing before giving up
        if _is_envelope_shaped(payload):
            return payload
    return None


def coerce_issues(value: Any) -> list[str]:
    """Best-effort list of issue strings from whatever the model put in ``issues``."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [
            str(item).strip()
            for item in value
            if isinstance(item, (str, int, float)) and str(item).strip()
        ]
    return []


def read_envelope_prose(payload: Mapping[str, Any]) -> tuple[str, list[str], bool | None]:
    """Split an envelope into ``(summary, issues, complete)``.

    Each field is read defensively: the worker keeps whatever the model wrote, because
    the summary is what the next planning cycle reads, but a wrong-typed field becomes
    a default rather than an exception that loses the whole reply.
    """
    raw_summary = payload.get("summary")
    summary = raw_summary.strip() if isinstance(raw_summary, str) else ""
    issues = coerce_issues(payload.get("issues"))
    raw_complete = payload.get("complete")
    complete = raw_complete if isinstance(raw_complete, bool) else None
    return summary, issues, complete


# --- Safety validation -----------------------------------------------------------------


@dataclass(frozen=True)
class WriteRecord:
    """What happened to one requested file, for ``WorkerAgent.applied`` and the summary."""

    path: str
    action: str
    bytes: int = 0
    backup_path: str | None = None
    reason: str = ""


def _blocked_path_reason(parts: Sequence[str]) -> str | None:
    """Why ``parts`` may not be written, or ``None`` when the path is allowed."""
    for part in parts:
        lowered = part.lower()
        if lowered in FORBIDDEN_DIR_NAMES:
            return f"writes into {part}/ are not allowed"
        if lowered.endswith(FORBIDDEN_DIR_SUFFIXES):
            return f"writes into {part}/ are not allowed"
    if parts and parts[-1].lower().endswith(FORBIDDEN_FILE_SUFFIXES):
        return f"{parts[-1]} is generated bytecode and is never written"
    return None


def _workspace_root(workspace: Path) -> Path:
    """The workspace as an absolute, symlink-resolved directory."""
    return Path(workspace).expanduser().resolve()


def resolve_write_path(raw: object, workspace: Path) -> Path:
    """Resolve a model-supplied path to an absolute destination inside the workspace.

    Raises ``ValueError`` with the reason for every rejection, because this is the one
    check that must fail closed: a clamp or a sanitised fallback would still write a
    model-chosen file somewhere. Rejected are absolute paths (POSIX and Windows-drive),
    ``..`` traversal, anything landing in a protected directory or generated bytecode,
    anything whose *resolved* location escapes the workspace (which is what catches a
    symlink pointing out of it), and a path that already exists as a directory.
    """
    if not isinstance(raw, str):
        raise ValueError(f"path must be a string, got {type(raw).__name__}")
    text = raw.strip()
    if not text:
        raise ValueError("path is empty")
    if "\x00" in text:
        raise ValueError("path contains a NUL byte")

    candidate = Path(text)
    if candidate.is_absolute() or PureWindowsPath(text).is_absolute() or candidate.drive:
        raise ValueError(f"path must be relative to the workspace, got {text!r}")
    if text.startswith("~"):
        raise ValueError(f"path must be relative to the workspace, got {text!r}")
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"path escapes the workspace with '..': {text!r}")
    blocked = _blocked_path_reason(candidate.parts)
    if blocked is not None:
        raise ValueError(blocked)

    root = _workspace_root(workspace)
    target = (root / candidate).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path resolves outside the workspace: {text!r} -> {target}")
    if target.is_dir():
        raise ValueError(f"path is an existing directory: {text!r}")
    return target


def workspace_relative(target: Path, workspace: Path) -> str:
    """The POSIX path of ``target`` relative to the workspace root.

    ``resolve_write_path`` has already proved containment, so the ``ValueError`` branch
    is defensive only; the bare name keeps a record readable rather than empty.
    """
    try:
        return target.relative_to(_workspace_root(workspace)).as_posix()
    except ValueError:  # pragma: no cover - unreachable after resolve_write_path
        return target.name


# --- Backups and atomic writes ----------------------------------------------------------


def backup_run_dir(backup_dir: Path, task_id: str, stamp: datetime | None = None) -> Path:
    """``<backup_dir>/<UTC timestamp>-<task id>``, the directory for one task's backups."""
    moment = stamp if stamp is not None else datetime.now(timezone.utc)
    safe_id = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(task_id))
    return Path(backup_dir).expanduser() / f"{moment.strftime(BACKUP_TIME_FORMAT)}-{safe_id or 'task'}"


def resolve_backup_dir(backup_dir: Path, workspace: Path) -> Path:
    """Anchor a relative ``backup_dir`` inside the workspace.

    The default ``.furrow/backups`` is relative, and the process CWD is whatever the
    user launched furrow from - which is not necessarily the workspace being edited.
    Anchoring it keeps the backups next to the code they restore instead of scattering
    them across the launch directory; an absolute ``backup_dir`` is honoured as given.
    """
    path = Path(backup_dir).expanduser()
    return path if path.is_absolute() else _workspace_root(workspace) / path


def create_backup(source: Path, relative_path: str, destination_dir: Path) -> Path:
    """Copy ``source`` to ``destination_dir/relative_path`` and return the new path.

    The original bytes are copied before the destination is replaced, so the backup is
    always the file as it was. An existing backup is never clobbered: a numeric suffix
    is added until ``open(..., "xb")`` - exclusive creation - succeeds, which also makes
    two backups of the same path in the same second distinct files instead of a race.
    Any failure propagates as ``OSError``; the caller refuses the overwrite.
    """
    target = Path(destination_dir) / relative_path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"could not create backup directory {target.parent}: {exc}") from exc
    for attempt in range(1, MAX_BACKUP_ATTEMPTS + 1):
        candidate = target if attempt == 1 else target.with_name(f"{target.name}-{attempt}")
        try:
            handle = candidate.open("xb")
        except FileExistsError:
            continue
        try:
            with handle, source.open("rb") as original:
                shutil.copyfileobj(original, handle)
        except BaseException:
            candidate.unlink(missing_ok=True)
            raise
        return candidate
    raise OSError(f"no free backup name for {target} after {MAX_BACKUP_ATTEMPTS} attempts")


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` so no partial file is ever observable.

    The content goes to a temporary file in the destination directory and is then moved
    into place with ``os.replace``, which is atomic within a filesystem: a reader sees
    either the old file or the whole new one. The temp file is removed if any step fails.

    ``LLMClient.write_file`` is a plain truncating write and cannot be used here - a
    process killed mid-write would leave half a source file behind - so the two-step
    write lives in the worker, called through ``asyncio.to_thread`` so the blocking
    ``fsync``/``replace`` do not stall the event loop that the orchestrator runs all
    workers on.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=TEMP_PREFIX, suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A replaced file keeps its own mode - an overwritten executable script must
            # stay executable - and a new file gets the usual world-readable 0644 instead
            # of mkstemp's private 0600, which would make it unreadable for anyone but
            # this process.
            shutil.copymode(path, temp_path)
        except OSError:
            temp_path.chmod(0o644)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


# --- Applying an envelope ---------------------------------------------------------------


async def apply_envelope_files(
    payload: Mapping[str, Any],
    *,
    workspace: Path,
    settings: Settings,
    task_id: str,
    stamp: datetime | None = None,
) -> list[WriteRecord]:
    """Validate and write the ``files`` entries of an envelope; return one record each.

    Every entry produces exactly one :class:`WriteRecord`, including the rejected ones:
    an edit the model asked for and the workspace refused must be visible to the next
    planning cycle, never silently dropped. Ordering of the checks is deliberate -
    shape, then size, then destination, then budget, then disk - so the recorded reason
    names the first thing that was actually wrong.
    """
    records: list[WriteRecord] = []
    raw_files = payload.get("files")
    if raw_files is None:
        return records
    if not isinstance(raw_files, list):
        records.append(
            WriteRecord(
                path="files",
                action=ACTION_SKIPPED,
                reason=f"'files' must be a JSON array, got {type(raw_files).__name__}",
            )
        )
        return records

    backup_root = resolve_backup_dir(settings.backup_dir, workspace)
    # A relative ``backup_dir`` is anchored inside the workspace, so backups are
    # themselves writable by default. ``atomic_write_text`` ends in ``os.replace``,
    # which overwrites unconditionally, so without this guard the recovery path sits
    # inside the blast radius of the reply that is supposed to be bounded by it: a
    # model that names a backup file would destroy the only restorable copy. When
    # ``backup_dir`` is the workspace root itself there is no separate tree to
    # protect, and rejecting every write would be worse than leaving it alone.
    protect_backups = backup_root != _workspace_root(workspace)
    run_dir = backup_run_dir(backup_root, task_id, stamp)
    written = 0
    total = 0

    for index, entry in enumerate(raw_files):
        label = f"files[{index}]"
        if not isinstance(entry, dict):
            records.append(
                WriteRecord(
                    path=label, action=ACTION_SKIPPED, reason="edit is not a JSON object"
                )
            )
            continue

        raw_path = entry.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            label = clamp_head(raw_path.strip(), MAX_PATH_CHARS)

        action = entry.get("action", ACTION_WRITE)
        if not isinstance(action, str) or action.strip().lower() != ACTION_WRITE:
            records.append(
                WriteRecord(
                    path=label,
                    action=ACTION_SKIPPED,
                    reason=f"unsupported action {action!r}; only {ACTION_WRITE!r} is supported",
                )
            )
            continue

        content = entry.get("content")
        if not isinstance(content, str):
            records.append(
                WriteRecord(
                    path=label,
                    action=ACTION_SKIPPED,
                    reason=f"'content' must be a string, got {type(content).__name__}",
                )
            )
            continue

        try:
            size = len(content.encode("utf-8"))
        except UnicodeEncodeError as exc:
            # ``json.loads`` turns "\ud800" into a lone surrogate, which no UTF-8
            # encoder accepts. Letting that escape here would abort the whole
            # envelope from inside the write, throwing away the records for files
            # already written and skipping every edit after this one.
            records.append(
                WriteRecord(
                    path=label,
                    action=ACTION_SKIPPED,
                    reason=f"'content' is not valid UTF-8 text ({exc})",
                )
            )
            continue

        try:
            target = resolve_write_path(raw_path, workspace)
        except ValueError as exc:
            records.append(
                WriteRecord(path=label, action=ACTION_SKIPPED, reason=str(exc))
            )
            continue

        if protect_backups and (target == backup_root or backup_root in target.parents):
            records.append(
                WriteRecord(
                    path=label,
                    action=ACTION_SKIPPED,
                    reason=(
                        "writes into the backup directory "
                        f"({workspace_relative(backup_root, workspace)}/) are not allowed"
                    ),
                )
            )
            continue

        if size > settings.max_write_bytes:
            records.append(
                WriteRecord(
                    path=label,
                    action=ACTION_SKIPPED,
                    bytes=size,
                    reason=(
                        f"{size} bytes exceeds max_write_bytes "
                        f"({settings.max_write_bytes})"
                    ),
                )
            )
            continue
        if written >= settings.max_files_per_task:
            records.append(
                WriteRecord(
                    path=label,
                    action=ACTION_SKIPPED,
                    bytes=size,
                    reason=(
                        f"exceeds max_files_per_task ({settings.max_files_per_task}) "
                        "for this task"
                    ),
                )
            )
            continue
        if total + size > settings.max_total_write_bytes:
            records.append(
                WriteRecord(
                    path=label,
                    action=ACTION_SKIPPED,
                    bytes=size,
                    reason=(
                        f"would exceed max_total_write_bytes "
                        f"({settings.max_total_write_bytes})"
                    ),
                )
            )
            continue

        relative = workspace_relative(target, workspace)
        existed = target.exists()

        if not settings.apply_changes:
            records.append(
                WriteRecord(
                    path=relative,
                    action=ACTION_SKIPPED,
                    bytes=size,
                    reason="apply_changes is disabled; reported only, nothing written",
                )
            )
            continue

        backup: Path | None = None
        if existed:
            try:
                backup = create_backup(target, relative, run_dir)
            except OSError as exc:
                # Fail closed: without a restorable copy of the original, an overwrite
                # is an unrecoverable edit to someone else's repository.
                logger.warning("Worker %s: refusing to overwrite %s: %s", task_id, relative, exc)
                records.append(
                    WriteRecord(
                        path=relative,
                        action=ACTION_SKIPPED,
                        bytes=size,
                        reason=f"refused to overwrite: backup failed ({type(exc).__name__}: {exc})",
                    )
                )
                continue

        try:
            await asyncio.to_thread(atomic_write_text, target, content)
        except OSError as exc:
            logger.warning("Worker %s: failed to write %s: %s", task_id, relative, exc)
            records.append(
                WriteRecord(
                    path=relative,
                    action=ACTION_SKIPPED,
                    bytes=size,
                    backup_path=str(backup) if backup else None,
                    reason=f"write failed ({type(exc).__name__}: {exc})",
                )
            )
            continue

        written += 1
        total += size
        records.append(
            WriteRecord(
                path=relative,
                action=ACTION_MODIFIED if existed else ACTION_CREATED,
                bytes=size,
                backup_path=str(backup) if backup else None,
            )
        )

    return records


# --- Result rendering ------------------------------------------------------------------


def format_record_line(record: WriteRecord) -> str:
    """One bounded human-readable line per applied file."""
    if record.action == ACTION_SKIPPED:
        return clamp_head(f"skipped {record.path}: {record.reason}", MAX_REPORT_LINE_CHARS)
    details = f"{record.bytes} bytes"
    if record.backup_path:
        details += f", backup: {record.backup_path}"
    return clamp_head(f"{record.action} {record.path} ({details})", MAX_REPORT_LINE_CHARS)


def format_report(
    summary: str,
    records: Sequence[WriteRecord],
    issues: Sequence[str] = (),
    complete: bool | None = None,
) -> str:
    """Render the worker's outcome as the task result the orchestrator stores.

    Rejections are rendered *before* the success list: a refused edit is the thing the
    next planning cycle must not lose, so it stays visible even when a verbose summary
    plus ten applied files push the result to the ``MAX_SUMMARY_CHARS`` cap. The model's
    own prose is the only thing that shrinks to make room.
    """
    applied = [r for r in records if r.action in (ACTION_CREATED, ACTION_MODIFIED)]
    skipped = [r for r in records if r.action == ACTION_SKIPPED]

    lines: list[str] = []
    if skipped:
        lines.append(f"Rejected or skipped {len(skipped)} requested edit(s):")
        lines.extend(format_record_line(record) for record in skipped)
    if applied:
        lines.append(f"Applied {len(applied)} file(s):")
        lines.extend(format_record_line(record) for record in applied)
    if issues:
        lines.append("Issues reported by the worker:")
        lines.extend(f"- {clamp_head(issue, MAX_ISSUE_CHARS)}" for issue in issues[:MAX_ISSUES])
        if len(issues) > MAX_ISSUES:
            lines.append(f"- ... and {len(issues) - MAX_ISSUES} further issues")
    if complete is not None:
        lines.append(f"Worker marked the task complete: {'yes' if complete else 'no'}")

    body = "\n".join(lines)
    head = clamp_head(summary, MAX_REPORT_SUMMARY_CHARS)
    report = f"{head}\n\n{body}" if body else head
    overflow = len(report) - MAX_SUMMARY_CHARS
    if overflow > 0:
        head = clamp_head(summary, max(0, len(head) - overflow))
        report = f"{head}\n\n{body}" if body else head
    return clamp_head(report, MAX_SUMMARY_CHARS)


class WorkerAgent:
    """Implements a single planned task, grounded in real repository content.

    The prompt is built from the task description, its dependencies and the actual text of
    the files it targets, so the model edits the code that exists instead of a guess.
    Context building never raises: a stale path, an unreadable file or a failing workspace
    listing each degrade to a note so the task still runs.

    The reply is a JSON envelope; its ``files`` entries are written to the workspace by
    :func:`apply_envelope_files` after every one of them passes the safety checks in
    :func:`resolve_write_path`, the size caps and - for an overwrite - a backup. What
    happened is readable afterwards on :attr:`applied`, one record per requested file
    including every rejection, and the same facts are rendered into the returned summary
    so the next planning cycle sees refused edits rather than assuming success.

    Args:
        task: The planned task to implement.
        client: LLM client to use; a default one is built from ``settings`` when omitted.
        settings: Used only to build the default client.
        dependency_results: Optional ``{task id: result}`` mapping for completed
            dependencies. Optional so the orchestrator's existing
            ``WorkerAgent(task=..., client=...)`` call site keeps working unchanged.
            May also be passed per call to :meth:`run`.
    """

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        dependency_results: Mapping[str, Any] | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        # Mirror the client's settings on the agent so model resolution follows the client
        # that will actually be called, whether it was injected or built here.
        self.settings = self.client.settings
        self.dependency_results: dict[str, Any] = dict(dependency_results or {})
        # Outcome of the last run(), rewritten from scratch on every run().
        self.applied: list[WriteRecord] = []
        self.advisory: bool = False

    async def run(self, dependency_results: Mapping[str, Any] | None = None) -> str:
        """Execute the task, apply the reply's edits and return a bounded summary.

        ``await worker.run()`` with no arguments is the whole contract the orchestrator
        uses; ``dependency_results`` here only overrides the constructor mapping for a
        single call, for callers that learn their dependencies' results after construction.

        A reply with no envelope is not an error: it becomes an advisory summary, nothing
        is written, :attr:`advisory` is set and :attr:`applied` carries the reason. The
        returned text is the model's own prose in that case, unchanged, because a worker
        that decided to change nothing has nothing to add to what it said.
        """
        self.applied = []
        self.advisory = False
        deps = self.dependency_results if dependency_results is None else dict(dependency_results)
        context = await build_context(self.task, self.client, self.settings.workspace)
        prompt = build_prompt(self.task, context, deps)
        try:
            text = await self.client.complete(prompt, model=self.settings.worker_model)
        except asyncio.CancelledError:
            # Cooperative cancellation has to reach the orchestrator's gather() untouched.
            raise
        except Exception as exc:
            # Re-raised with the task and model named: the orchestrator stores str(exc) as
            # the task result, so this message is all the next planning cycle sees.
            raise WorkerError(
                f"worker task {self.task.id} failed to get a result from model "
                f"{self.settings.worker_model}: {type(exc).__name__}: {exc}"
            ) from exc

        payload = extract_edit_envelope(text)
        if payload is None:
            self.advisory = True
            self.applied.append(
                WriteRecord(
                    path="(worker reply)",
                    action=ACTION_SKIPPED,
                    reason="reply carried no JSON edit envelope; nothing was applied",
                )
            )
            return format_summary(text)

        summary, issues, complete = read_envelope_prose(payload)
        self.applied = await apply_envelope_files(
            payload,
            workspace=self.settings.workspace,
            settings=self.settings,
            task_id=self.task.id,
        )
        return format_summary(format_report(summary, self.applied, issues, complete))
