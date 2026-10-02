"""Bounded ``read``/``list``/``write``/``finish`` tool loop for a single task.

A worker is the part of Furrow that actually changes the repository, so the rules here
are about never losing the run:

* every action is sandboxed through :func:`furrow.util.resolve_within`, so a model that
  asks for ``../../etc/passwd`` gets an error back instead of a written file;
* every action failure is *recorded and fed back* rather than raised, because a single
  bad turn must not abort work that is already on disk;
* the loop is hard-bounded by ``worker_max_turns`` and always returns an honest,
  non-empty summary -- including when it ran out of turns, which is never reported as
  success.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_ACTION_SCHEMA, WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.util import JSONExtractionError, extract_json, resolve_within, truncate

if TYPE_CHECKING:
    from furrow.config import Settings
    from furrow.core.events import EventCallback

__all__ = ["WorkerAgent"]

_READ_LIMIT = 2000
_LIST_LIMIT = 1500
_LIST_FILES = 60
_TASK_LIMIT = 4000
_ECHO_LIMIT = 500
_RESULT_TURNS = 3
_ERROR_TURNS = 5
_DEFAULT_MAX_TURNS = 12
_MISSING_KEY = "action requires"


class WorkerAgent:
    """Implements one :class:`~furrow.config.TaskModel` with a bounded tool loop."""

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client if client is not None else LLMClient(settings=settings)
        self.settings: Settings = settings if settings is not None else self.client.settings
        self.actions: list[dict[str, Any]] = []
        self.files_written: list[str] = []
        self.files_read: list[str] = []
        self.errors: list[str] = []
        self.turns: int = 0
        self._results: list[str] = []

    async def run(self, *, on_event: EventCallback | None = None) -> str:
        """Run the tool loop until the model finishes or the turn budget runs out.

        Args:
            on_event: Optional progress callback; at most one ``LOG`` event is emitted
                per action taken.

        Returns:
            A non-empty summary of what happened.  It describes only what really
            happened: a turn limit or an error is never dressed up as success.
        """
        max_turns = self._max_turns()
        while self.turns < max_turns:
            self.turns += 1
            raw = _as_text(await self._complete(self._turn_prompt()))
            if not raw.strip():
                return self._summary(
                    f"Worker stopped on turn {self.turns} of {max_turns}: "
                    f"{self.errors[-1] if self.errors else 'the model returned no output'}."
                )
            payload = self._parse_turn(raw)
            if payload is None:
                # A prose-only reply is treated as the worker having narrated its work.
                return raw.strip() or self._summary("Worker returned no output.")
            result = await self._dispatch(payload, raw, on_event)
            if result is not None:
                return result
        return self._summary(
            f"Worker hit the {max_turns}-turn limit before finishing, so its work is "
            "incomplete and unverified."
        )

    async def _complete(self, prompt: str) -> str:
        """One model call; a provider failure is recorded, never raised."""
        try:
            return await self.client.complete(
                prompt,
                system=WORKER_PROMPT,
                model=getattr(self.client.settings, "worker_model", None),
            )
        except Exception as exc:
            self._record_error(f"model call failed: {exc}")
            return ""

    def _parse_turn(self, raw: str) -> dict[str, Any] | None:
        """Return the action object for this turn, or ``None`` when the reply is prose."""
        try:
            data = extract_json(raw)
        except JSONExtractionError:
            return None
        if not isinstance(data, dict):
            return None
        return data

    async def _dispatch(
        self,
        payload: dict[str, Any],
        raw: str,
        on_event: EventCallback | None,
    ) -> str | None:
        """Run one action; return a summary when the action ends the run, else ``None``."""
        action = str(payload.get("action") or "").strip().lower()
        record: dict[str, Any] = {
            "turn": self.turns,
            "action": action or None,
            "path": _as_text(payload.get("path")).strip() or None,
            "ok": False,
        }
        self.actions.append(record)

        if action == "finish":
            summary = _as_text(payload.get("summary")).strip()
            record["ok"] = bool(summary)
            if summary:
                return summary
            self._record_error("finish action carried no 'summary'")
            return truncate(raw.strip(), limit=_ECHO_LIMIT) or self._summary(
                "Worker finished without a summary."
            )
        if action == "read":
            await self._do_read(payload, record, on_event)
        elif action == "list":
            await self._do_list(payload, record, on_event)
        elif action == "write":
            await self._do_write(payload, record, on_event)
        else:
            self._record_error(f"unknown or missing action {action or '(none)'!r}")
        return None

    async def _do_read(
        self,
        payload: dict[str, Any],
        record: dict[str, Any],
        on_event: EventCallback | None,
    ) -> None:
        path = _as_text(payload.get("path")).strip()
        if not path:
            self._record_error(f"read {_MISSING_KEY} a 'path'")
            return
        root = self._root()
        try:
            target = resolve_within(root, path)
            content = await self.client.read_file(target)
        except Exception as exc:
            self._record_error(f"read {path} failed: {exc}")
            return
        relative = _relative(target, root)
        self.files_read.append(relative)
        record["ok"] = True
        self._results.append(f"[read {relative}]\n{truncate(content, limit=_READ_LIMIT)}")
        await self._log(on_event, f"read {relative}", {"action": "read", "path": relative})

    async def _do_list(
        self,
        payload: dict[str, Any],
        record: dict[str, Any],
        on_event: EventCallback | None,
    ) -> None:
        raw_path = _as_text(payload.get("path")).strip()
        directory: Path | None = None
        if raw_path:
            try:
                directory = resolve_within(self._root(), raw_path)
            except Exception as exc:
                self._record_error(f"list {raw_path} failed: {exc}")
                return
        try:
            files = self.client.list_files(directory, max_files=_LIST_FILES)
        except Exception as exc:
            self._record_error(f"list {raw_path or '.'} failed: {exc}")
            return
        listing = truncate("\n".join(str(name) for name in files), limit=_LIST_LIMIT)
        record["ok"] = True
        self._results.append(f"[list {raw_path or '.'}]\n{listing or '(no files)'}")
        await self._log(
            on_event, f"listed {raw_path or '.'}", {"action": "list", "path": raw_path or "."}
        )

    async def _do_write(
        self,
        payload: dict[str, Any],
        record: dict[str, Any],
        on_event: EventCallback | None,
    ) -> None:
        path = _as_text(payload.get("path")).strip()
        if not path:
            self._record_error(f"write {_MISSING_KEY} a 'path'")
            return
        content = payload.get("content")
        if not isinstance(content, str):
            # A missing "content" would silently truncate the target file to nothing.
            self._record_error(
                f"write {path} {_MISSING_KEY} a 'content' string holding the full file"
            )
            return
        root = self._root()
        try:
            target = resolve_within(root, path)
            await self.client.write_file(target, content)
        except Exception as exc:
            self._record_error(f"write {path} failed: {exc}")
            return
        relative = _relative(target, root)
        self.files_written.append(relative)
        record["ok"] = True
        self._results.append(f"[write {relative}] wrote {len(content)} character(s)")
        await self._log(
            on_event, f"wrote {relative}", {"action": "write", "files": 1, "path": relative}
        )

    async def _log(
        self, on_event: EventCallback | None, message: str, data: dict[str, Any]
    ) -> None:
        """Emit the single LOG event that reports an action."""
        events = _events()
        await events.emit(
            on_event,
            events.make_event(
                events.EventKind.LOG,
                message=message,
                task_id=self.task.id,
                data=data,
            ),
        )

    def _turn_prompt(self) -> str:
        declared = (
            ", ".join(self.task.files) if self.task.files else "none declared - choose them yourself"
        )
        parts = [
            f"Task id: {self.task.id}",
            f"Task: {truncate(self.task.description, limit=_TASK_LIMIT)}",
            f"Declared files: {declared}",
            f"Repository root: {self._root()}",
            f"Turn {self.turns} of {self._max_turns()}.",
        ]
        if self._results:
            recent = "\n\n".join(self._results[-_RESULT_TURNS:])
            parts.append(f"Results of your most recent actions:\n{recent}")
        if self.errors:
            errors = "\n".join(f"- {item}" for item in self.errors[-_ERROR_TURNS:])
            parts.append(f"Errors so far (do not repeat them):\n{errors}")
        parts.append(f"Reply with exactly one JSON object:\n{WORKER_ACTION_SCHEMA}")
        return "\n".join(parts)

    def _record_error(self, message: str) -> None:
        self.errors.append(message)

    def _summary(self, headline: str) -> str:
        """Describe what this worker really did; never claim more than happened."""
        parts = [headline]
        if self.files_written:
            parts.append(f"Wrote {len(self.files_written)} file(s): {_listed(self.files_written)}.")
        if self.files_read:
            parts.append(f"Read {len(self.files_read)} file(s): {_listed(self.files_read)}.")
        if self.errors:
            parts.append(f"{len(self.errors)} error(s), first: {self.errors[0]}")
        if len(parts) == 1:
            parts.append("No files were read or written.")
        return " ".join(parts)

    def _root(self) -> Path:
        return Path(getattr(self.settings, "workspace", "."))

    def _max_turns(self) -> int:
        raw = getattr(self.settings, "worker_max_turns", _DEFAULT_MAX_TURNS)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return _DEFAULT_MAX_TURNS


def _events() -> ModuleType:
    """Import ``furrow.core.events`` on first use.

    ``furrow.core`` re-exports the orchestrator, and the orchestrator imports this
    package, so a module-level import here would be circular; by the time a worker
    actually runs, the graph is already resolved.
    """
    from furrow.core import events

    return events


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _relative(target: Path, root: Path) -> str:
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return str(target)


def _listed(paths: list[str]) -> str:
    shown = ", ".join(paths[:10])
    if len(paths) > 10:
        shown += f", ... (+{len(paths) - 10} more)"
    return shown
