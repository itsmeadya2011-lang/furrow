from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


MAX_TURNS = 4
MAX_READ_CHARS = 8000
MAX_LIST_ENTRIES = 200
VALID_ACTIONS = ("read", "write", "list", "done")


WORKER_TOOL_PROMPT = """You act on the workspace through tools. Emit exactly ONE JSON action
object per turn; the result of that action is handed back to you before your next reply.

Available actions:
1. {"action": "read", "path": "<relative path>"} - read a file; content comes back next turn.
2. {"action": "write", "path": "<relative path>", "content": "<complete file content>"} - create
   or overwrite a file (parent directories are created automatically).
3. {"action": "list", "path": "<relative directory>"} - list files under a directory; "." (the
   default) lists the workspace root.
4. {"action": "done", "summary": "<one or two sentences on what you changed>"} - finish the task.

Protocol rules:
- Reply with the JSON object only: no prose, no markdown fences, no multiple actions at once.
- Paths are relative to the workspace root. Absolute paths are accepted only when they stay inside
  the workspace root; anything that escapes it is rejected.
- Writes only target files, never directories.
- Read a file before overwriting it, unless you are creating it from scratch.
- Writes replace the whole file, so send the complete new content.
- When the task is done (or you are blocked), reply with action "done" and explain why.
"""


@dataclass
class _Action:
    action: str
    path: str | None = None
    content: str | None = None
    summary: str = ""


@dataclass
class _Outcome:
    """Result of executing one action: either the loop is finished, or feedback for the next turn."""

    feedback: str
    done: bool = False
    summary: str = ""


@dataclass
class _Record:
    read: list[str] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.record = _Record()

    @property
    def workspace(self) -> Path:
        return Path(self.client.settings.workspace).expanduser().resolve()

    async def run(self) -> str:
        self.record = _Record()
        feedback = "Start now: reply with your first JSON action object."
        completed = False
        done_summary = ""
        turns_used = 0

        for turn in range(1, MAX_TURNS + 1):
            turns_used = turn
            prompt = self._build_prompt(feedback, turn)
            reply = await self.client.complete(prompt, model=self.client.settings.worker_model)
            parsed = self._parse(reply)
            if isinstance(parsed, str):
                # A reply we could not parse is a real failure of the cycle: record it
                # so it reaches the final summary, exactly like an execution failure.
                outcome = self._fail(parsed)
            else:
                outcome = await self._execute(parsed)
            if outcome.done:
                completed = True
                done_summary = outcome.summary
                break
            feedback = outcome.feedback

        return self._summarize(completed=completed, done_summary=done_summary, turns_used=turns_used)

    # ------------------------------------------------------------------ prompting

    def _build_prompt(self, feedback: str, turn: int) -> str:
        files = ", ".join(self.task.files) if self.task.files else "any (pick the files the task needs)"
        return (
            f"{WORKER_PROMPT}\n\n{WORKER_TOOL_PROMPT}\n"
            f"Task: {self.task.description}\n"
            f"Declared files: {files}\n"
            f"Workspace root: {self.workspace}\n"
            f"Turn {turn} of {MAX_TURNS}\n\n"
            f"Result of your previous action:\n{feedback}\n\n"
            f"Reply with the next single JSON action object."
        )

    # ------------------------------------------------------------------ parsing

    def _parse(self, reply: str) -> _Action | str:
        """Turn a raw LLM reply into an _Action, or return an error string to feed back."""
        text = (reply or "").strip()
        if not text:
            return "empty reply. Reply with a single JSON action object."

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end < start:
            return f"reply was not JSON: {text[:200]!r}"
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            return f"invalid JSON ({exc.msg}). Reply with one JSON action object, no prose or fences."

        if not isinstance(payload, dict):
            return 'JSON action must be an object with an "action" key.'

        action = payload.get("action")
        if not isinstance(action, str) or not action.strip():
            return f'missing "action" key. Use one of: {", ".join(VALID_ACTIONS)}.'
        action = action.strip().lower()
        if action not in VALID_ACTIONS:
            return f'unknown action "{action}". Use one of: {", ".join(VALID_ACTIONS)}.'

        path = payload.get("path")
        if path is not None and not isinstance(path, str):
            return '"path" must be a string.'

        content = payload.get("content")
        if content is not None and not isinstance(content, str):
            content = json.dumps(content)

        summary = payload.get("summary")
        if not isinstance(summary, str):
            summary = "" if summary is None else str(summary)

        return _Action(action=action, path=path, content=content, summary=summary.strip())

    # ------------------------------------------------------------------ path guard

    def _resolve(self, raw: str | None) -> Path | None:
        """Resolve a model-supplied path inside the workspace, or None if it escapes/is empty."""
        root = self.workspace
        if not raw or not raw.strip():
            return None
        candidate = Path(raw.strip()).expanduser()
        try:
            resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        except OSError:
            return None
        if not resolved.is_relative_to(root):
            return None
        return resolved

    def _display(self, path: Path) -> str:
        root = self.workspace
        if path == root:
            return "."
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return str(path)

    def _fail(self, message: str) -> _Outcome:
        self.record.errors.append(message)
        return _Outcome(feedback=f"ERROR: {message}")

    # ------------------------------------------------------------------ execution

    async def _execute(self, action: _Action) -> _Outcome:
        if action.action == "done":
            summary = action.summary.strip() or "Worker reported done without a summary."
            return _Outcome(feedback="", done=True, summary=summary)
        if action.action == "read":
            return await self._do_read(action)
        if action.action == "write":
            return await self._do_write(action)
        if action.action == "list":
            return self._do_list(action)
        return self._fail(f'unknown action "{action.action}". Use one of: {", ".join(VALID_ACTIONS)}.')

    async def _do_read(self, action: _Action) -> _Outcome:
        resolved = self._resolve(action.path)
        if resolved is None:
            return self._fail(
                f'read requires a "path" inside the workspace root ({self.workspace}); '
                f'got {action.path!r}.'
            )
        name = self._display(resolved)
        if resolved.is_dir():
            return self._fail(f"{name} is a directory; use action \"list\" instead.")
        try:
            content = await self.client.read_file(resolved)
        except Exception as exc:  # noqa: BLE001 - never let one bad file kill the cycle
            return self._fail(f"could not read {name}: {exc}")
        if name not in self.record.read:
            self.record.read.append(name)
        if len(content) > MAX_READ_CHARS:
            content = content[:MAX_READ_CHARS] + f"\n... [truncated at {MAX_READ_CHARS} chars]"
        return _Outcome(feedback=f"Contents of {name}:\n```\n{content}\n```")

    async def _do_write(self, action: _Action) -> _Outcome:
        resolved = self._resolve(action.path)
        if resolved is None:
            return self._fail(
                f'write requires a "path" inside the workspace root ({self.workspace}); '
                f'got {action.path!r}.'
            )
        name = self._display(resolved)
        if resolved.is_dir() or name == ".":
            return self._fail(f"{name} is a directory; writes are only allowed to files.")
        if action.content is None:
            return self._fail(f'write to {name} is missing a "content" field with the complete file content.')
        existed = resolved.exists() and name in self.record.read
        try:
            await self.client.write_file(resolved, action.content)
        except Exception as exc:  # noqa: BLE001 - never let one bad file kill the cycle
            return self._fail(f"could not write {name}: {exc}")
        if name not in self.record.written:
            self.record.written.append(name)
        note = " (overwrote an existing file)" if existed else ""
        return _Outcome(feedback=f"Wrote {len(action.content)} chars to {name}{note}.")

    def _do_list(self, action: _Action) -> _Outcome:
        raw = action.path.strip() if action.path and action.path.strip() else "."
        resolved = self._resolve(raw)
        if resolved is None:
            return self._fail(f"cannot list {raw!r}: path escapes the workspace root ({self.workspace}).")
        name = self._display(resolved)
        try:
            entries = sorted(self.client.list_files(resolved))
        except Exception as exc:  # noqa: BLE001 - never let one bad listing kill the cycle
            return self._fail(f"could not list {name}: {exc}")
        shown = entries[:MAX_LIST_ENTRIES]
        body = "\n".join(shown) if shown else "(no files)"
        if len(entries) > MAX_LIST_ENTRIES:
            body += f"\n... [{len(entries) - MAX_LIST_ENTRIES} more files not shown]"
        return _Outcome(feedback=f"Files under {name}:\n{body}")

    # ------------------------------------------------------------------ summary

    def _summarize(self, *, completed: bool, done_summary: str, turns_used: int) -> str:
        read = ", ".join(self.record.read) or "none"
        written = ", ".join(self.record.written) or "none"
        lines = [
            f"Task {self.task.id}: {self.task.description}",
            done_summary.strip() or "Worker stopped without a summary.",
            f"Files written: {written}",
            f"Files read: {read}",
        ]
        if not completed:
            lines.append(f"Worker stopped at the turn cap ({turns_used}/{MAX_TURNS}) without action 'done'.")
        if self.record.errors:
            lines.append("Errors: " + "; ".join(self.record.errors))
        return "\n".join(lines)