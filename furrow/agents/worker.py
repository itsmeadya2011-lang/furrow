"""Worker agent: implements one task and applies the edits it returns.

This is where the loop actually changes the codebase. The model replies with a
``WorkerResult`` JSON document listing file edits; the worker validates it and applies
each edit through :class:`~furrow.llm.LLMClient`, which keeps every write atomic and
confined to ``settings.workspace``. A single bad edit is recorded and the rest are
still applied -- an edit that cannot be trusted must not cost the whole task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from furrow.agents.jsonutil import JSONExtractionError, extract_json_object, truncate_output
from furrow.agents.prompts import WORKER_PROMPT, WORKER_SYSTEM
from furrow.config import FileEdit, TaskModel, TaskStatus, WorkerResult
from furrow.llm import LLMClient, LLMError

if TYPE_CHECKING:
    from furrow.config import Settings

# Rough characters-per-token ratio for source code; keeps the prompt inside the
# per-request token budget configured by ``llm_max_tokens``.
_CHARS_PER_TOKEN = 2
_MAX_CONTEXT_CHARS = 24_000
_MAX_LISTING = 200
_MAX_CONTEXT_FILES = 6
_MAX_SUMMARY_CHARS = 4_000


class WorkerAgent:
    """Executes a single :class:`TaskModel` by applying model-proposed file edits.

    The agent owns no state beyond its task and client, and its ``settings`` always
    come from that client so there is a single source of truth for configuration.
    """

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        # ``LLMClient`` resolves ``settings=None`` to ``get_settings()``, so passing
        # None through explicitly is safe. (The old client stored None verbatim, which
        # made every later ``self.client.settings.worker_model`` an AttributeError.)
        self.client = client or LLMClient(settings=settings)

    @property
    def settings(self) -> Settings:
        """The settings used for working, read from the client."""
        return self.client.settings

    async def run(self) -> WorkerResult:
        """Implement the task and return the edits that were applied.

        Returns a :class:`WorkerResult`; ``str(result)`` is the summary, so callers
        written against the old ``str`` return type keep working.
        """
        prompt = await self._build_prompt()
        response = await self.client.complete(
            prompt,
            model=self.settings.worker_model,
            system=WORKER_SYSTEM,
        )
        return await self._apply_response(response)

    # -- prompt construction ------------------------------------------------

    async def _build_prompt(self) -> str:
        """Assemble the user message, including the code the model has to anchor on."""
        declared = [f for f in self.task.files if f.strip()]
        sections = [
            WORKER_PROMPT,
            f"\nTask id: {self.task.id}",
            f"\nTask description:\n{self.task.description}",
        ]
        if declared:
            sections.append(f"\nFiles declared for this task: {', '.join(declared)}")
        else:
            sections.append(
                "\nFiles declared for this task: (none listed - choose what the task needs)"
            )
        if self.task.dependencies:
            sections.append(f"Depends on tasks: {', '.join(self.task.dependencies)}")

        listing = await self._workspace_listing()
        if listing:
            sections.append(f"\nWorkspace files ({len(listing)} shown):\n" + "\n".join(listing))

        context = await self._file_context(declared)
        if context:
            sections.append(
                "\nCurrent contents of the declared files. An 'edit' anchor must be copied"
                " from this text character for character, including indentation, and must"
                " appear exactly once in its file. A file listed as unreadable below does"
                " not exist yet - create it with action 'write'.\n\n" + context
            )
        return "\n".join(sections)

    async def _workspace_listing(self) -> list[str]:
        """Return a bounded listing of the workspace, or ``[]`` if it is unreadable."""
        try:
            files = await self.client.list_files(
                root=self.settings.workspace, max_results=_MAX_LISTING
            )
        except (LLMError, OSError):
            return []
        return sorted(files)[:_MAX_LISTING]

    async def _file_context(self, declared: list[str]) -> str:
        """Return the current text of the declared files, within a character budget."""
        if not declared:
            return ""
        max_chars = self.settings.llm_max_tokens * _CHARS_PER_TOKEN
        budget = max(2_000, min(_MAX_CONTEXT_CHARS, max_chars))
        targets = declared[:_MAX_CONTEXT_FILES]
        per_file = max(400, budget // len(targets))
        chunks: list[str] = []
        used = 0
        for path in targets:
            if used >= budget:
                chunks.append(f"--- {path} ---\n(budget exhausted; not shown)")
                continue
            try:
                content = await self.client.read_file(path, root=self.settings.workspace)
            except (LLMError, OSError) as exc:
                chunks.append(f"--- {path} ---\n(unreadable: {exc})")
                continue
            shown = truncate_output(content, per_file)
            chunks.append(f"--- {path} ---\n{shown}")
            used += len(shown)
        return "\n\n".join(chunks)

    # -- response handling --------------------------------------------------

    async def _apply_response(self, response: str) -> WorkerResult:
        """Validate the model reply and apply its edits, never raising on a bad edit."""
        try:
            data = extract_json_object(response)
        except JSONExtractionError:
            return self._unstructured(
                response,
                "model reply was not JSON; used as a prose summary with no edits applied",
            )
        try:
            result = WorkerResult.model_validate(data)
        except ValidationError as exc:
            summary = data.get("summary") if isinstance(data.get("summary"), str) else None
            return self._unstructured(
                summary or response,
                f"model reply did not match the WorkerResult schema ({exc.error_count()} "
                "problem(s)); no edits applied",
            )

        return await self._apply_edits(result)

    async def _apply_edits(self, result: WorkerResult) -> WorkerResult:
        """Apply every proposed edit, collecting per-edit outcomes into the result."""
        for edit in result.edits:
            error = await self._apply_edit(edit)
            if error is None:
                result.applied.append(edit.path)
            else:
                result.errors.append(error)
        if result.edits and not result.applied:
            # Nothing the model asked for landed: the task did not happen.
            result.status = TaskStatus.FAILED
        result.summary = self._compose_summary(result)
        return result

    async def _apply_edit(self, edit: FileEdit) -> str | None:
        """Apply one edit. Returns ``None`` on success, or an error string."""
        if not edit.path.strip():
            return "edit rejected: missing 'path'"
        try:
            if edit.action == "write":
                if edit.content is None:
                    return f"{edit.path}: action 'write' requires 'content'"
                await self.client.write_file(
                    edit.path, edit.content, root=self.settings.workspace
                )
                return None
            if not edit.search:
                return f"{edit.path}: action 'edit' requires a non-empty 'search' anchor"
            if edit.replace is None:
                return f"{edit.path}: action 'edit' requires 'replace'"
            original = await self.client.read_file(edit.path, root=self.settings.workspace)
            # str.count reports non-overlapping occurrences, which is the number of
            # anchors the edit would apply to.
            matches = original.count(edit.search)
            if matches == 0:
                return f"{edit.path}: 'search' anchor not found in the file"
            if matches > 1:
                return (
                    f"{edit.path}: 'search' anchor is ambiguous ({matches} matches); "
                    "it must appear exactly once"
                )
            await self.client.write_file(
                edit.path,
                original.replace(edit.search, edit.replace, 1),
                root=self.settings.workspace,
            )
            return None
        except (LLMError, OSError) as exc:
            return f"{edit.path}: {exc}"

    @staticmethod
    def _unstructured(text: str, note: str) -> WorkerResult:
        """Degrade gracefully when the reply is not the expected JSON document.

        A model that reports in prose has still told us something; crashing the cycle
        over a formatting quirk is strictly worse than recording a note and carrying on.
        """
        summary = (text or "").strip()
        if not summary:
            return WorkerResult(
                summary="Worker returned an empty response.",
                status=TaskStatus.FAILED,
                errors=[note],
            )
        return WorkerResult(
            summary=truncate_output(summary, _MAX_SUMMARY_CHARS),
            status=TaskStatus.COMPLETED,
            errors=[note],
        )

    @staticmethod
    def _compose_summary(result: WorkerResult) -> str:
        """Append applied paths and per-edit errors to the model-written summary."""
        notes: list[str] = []
        if result.applied:
            notes.append("Applied edits:\n" + "\n".join(f"- {path}" for path in result.applied))
        if result.errors:
            notes.append("Unapplied edits:\n" + "\n".join(f"- {e}" for e in result.errors))
        if not notes:
            return result.summary
        return f"{result.summary.rstrip()}\n\n" + "\n".join(notes)
