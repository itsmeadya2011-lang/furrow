"""The worker agent: the only component in Furrow that actually edits files.

``run()`` reads the current contents of the task's files, asks the model for a
strict JSON edit envelope, and writes each edit to disk.  There is deliberately
no tool-calling loop: the model sees the files once, replies with whole file
bodies, and they are applied verbatim.  That keeps the write surface auditable -
every byte written appears in the returned summary and in ``written_files``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient, extract_json

if TYPE_CHECKING:
    from furrow.config import Settings


# --- module level tunables -------------------------------------------------

#: Hard cap on how many assigned files are pulled into the prompt.  Beyond this
#: the remaining paths are named in the context block but their contents are not
#: sent, so a 200-file task cannot blow the context window.
MAX_CONTEXT_FILES = 20

#: Hard cap on the combined size of the file-context block, in characters.
#: The block is truncated to exactly this many characters and says so.
MAX_CONTEXT_CHARS = 20_000


WORKER_SYSTEM_PROMPT = (
    "You are a senior software engineer inside Furrow, an autonomous coding agent. "
    "You answer with a single JSON edit envelope and nothing else."
)

CONTEXT_HEADER = "=== FILE CONTEXT (current contents) ==="
CONTEXT_FOOTER = "=== END FILE CONTEXT ==="
FILE_BEGIN = "=== BEGIN FILE: {path} ==="
FILE_END = "=== END FILE: {path} ==="
NO_CONTEXT = "(no readable files were assigned to this task)"

TRUNCATION_NOTICE = (
    f"\n[truncated: the file context above was cut at the {MAX_CONTEXT_CHARS} "
    "character budget, so some content is missing]"
)

DEFAULT_SUMMARY = "worker returned edits without a summary"


def _fit_block(block: str) -> str:
    """Clamp *block* to ``MAX_CONTEXT_CHARS``, appending a notice when it bites."""
    if len(block) <= MAX_CONTEXT_CHARS:
        return block
    keep = max(0, MAX_CONTEXT_CHARS - len(TRUNCATION_NOTICE))
    return block[:keep] + TRUNCATION_NOTICE


class WorkerAgent:
    """Implements one :class:`~furrow.config.TaskModel` and writes the result."""

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        dry_run: bool = False,
    ) -> None:
        self.task = task
        self.settings = settings
        self.dry_run = dry_run
        #: Paths written by the most recent :meth:`run`, in order.  Stays empty
        #: under ``dry_run``.
        self.written_files: list[str] = []
        if client is not None:
            self.client = client
        elif settings is not None:
            self.client = LLMClient(settings=settings)
        else:
            # NOTE: never pass ``None`` here. ``LLMClient.__init__`` only falls back
            # to the global settings for an *omitted* argument; an explicit ``None``
            # would set ``client.settings = None`` and break every ``.settings.*``
            # lookup below.
            self.client = LLMClient()

    # --- prompt assembly -----------------------------------------------------

    async def _file_context(self) -> str:
        """Read the task's files and render them as a bounded, delimited block."""
        paths = list(self.task.files)
        omitted = paths[MAX_CONTEXT_FILES:]
        paths = paths[:MAX_CONTEXT_FILES]

        blocks: list[str] = []
        missing: list[str] = []
        for path in paths:
            try:
                content = await self.client.read_file(path)
            except (OSError, ValueError, TypeError):
                # A file that is missing, unreadable, a directory or not text is
                # not fatal: it is reported in the block as missing instead.
                missing.append(path)
                continue
            blocks.append(
                f"{FILE_BEGIN.format(path=path)}\n{content}\n{FILE_END.format(path=path)}"
            )

        lines: list[str] = [CONTEXT_HEADER]
        if not blocks and not missing:
            lines.append(NO_CONTEXT)
        if omitted:
            lines.append(
                f"[note: {len(omitted)} further assigned file(s) were omitted to stay "
                f"within the {MAX_CONTEXT_FILES} file limit: {', '.join(omitted)}]"
            )
        if missing:
            lines.append(
                "[note: these assigned files could not be read and may not exist: "
                f"{', '.join(missing)}]"
            )
        lines.extend(blocks)
        lines.append(CONTEXT_FOOTER)
        # The notes come first so they survive truncation of a huge context block.
        return _fit_block("\n".join(lines))

    def build_prompt(self, context: str) -> str:
        """Join the instructions, the task and the file context into one prompt."""
        targets = ", ".join(self.task.files) if self.task.files else "(none assigned)"
        return (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Assigned files: {targets}\n\n"
            f"{context}\n"
        )

    # --- reply parsing -------------------------------------------------------

    @staticmethod
    def _parse_envelope(reply: str) -> tuple[str, list[dict[str, str]]]:
        """Turn the raw model reply into ``(summary, edits)``.

        Raises ``ValueError`` quoting *reply* when the envelope is malformed.
        """

        def reject(reason: str) -> ValueError:
            return ValueError(f"malformed worker reply ({reason}); raw reply: {reply!r}")

        payload: Any = extract_json(reply)
        if not isinstance(payload, dict):
            raise reject(f"expected a JSON object, got {type(payload).__name__}")

        raw_summary = payload.get("summary")
        if isinstance(raw_summary, str) and raw_summary.strip():
            summary: str = raw_summary.strip()
        else:
            summary = DEFAULT_SUMMARY

        raw_edits = payload.get("edits", [])
        if not isinstance(raw_edits, list):
            raise reject(f"'edits' must be a list, got {type(raw_edits).__name__}")

        edits: list[dict[str, str]] = []
        for index, entry in enumerate(raw_edits):
            if not isinstance(entry, dict):
                raise reject(f"edit #{index} is {type(entry).__name__}, not an object")
            path = entry.get("path")
            content = entry.get("content")
            if not isinstance(path, str) or not path.strip():
                raise reject(f"edit #{index} has no usable string 'path'")
            if not isinstance(content, str):
                raise reject(
                    f"edit #{index} ('{path.strip()}') has a "
                    f"{type(content).__name__} 'content', expected the complete file text"
                )
            edits.append({"path": path.strip(), "content": content})
        return summary, edits

    # --- entry point ---------------------------------------------------------

    async def run(self) -> str:
        context = await self._file_context()
        prompt = self.build_prompt(context)

        reply = await self.client.complete(
            prompt,
            system=WORKER_SYSTEM_PROMPT,
            model=self.client.settings.worker_model,
        )
        summary, edits = self._parse_envelope(reply)

        self.written_files = []
        planned: list[str] = []
        for edit in edits:
            if not self.dry_run:
                await self.client.write_file(edit["path"], edit["content"])
                self.written_files.append(edit["path"])
            planned.append(edit["path"])

        wrote = ", ".join(planned) if planned else "(none)"
        return f"{summary}\nWrote: {wrote}"
