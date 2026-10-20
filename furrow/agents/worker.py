from __future__ import annotations

from pathlib import Path

from furrow.agents.parsing import LLMParseError, extract_json_object
from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

MAX_FILE_CONTEXT = 20_000
MAX_TOTAL_CONTEXT = 200_000
MAX_EDIT_SIZE = 2_000_000


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.task = task
        self.settings = settings
        self.client = client or LLMClient(settings=settings)
        effective_settings = settings
        if effective_settings is None:
            candidate = getattr(self.client, "settings", None)
            if isinstance(candidate, Settings):
                effective_settings = candidate
        if effective_settings is None:
            effective_settings = Settings()
        self.effective_settings = effective_settings
        self.workspace = (
            Path(workspace) if workspace is not None else self.effective_settings.workspace
        )
        try:
            self.workspace_resolved = self.workspace.resolve()
        except Exception:
            self.workspace_resolved = self.workspace
        self.files_changed: list[str] = []
        self.rejected_edits: list[str] = []

    def _build_context(self) -> str:
        parts: list[str] = []
        parts.append(f"Task description: {self.task.description}")
        parts.append(f"Workspace: {self.workspace_resolved}")
        files = list(self.task.files) if self.task.files else []
        parts.append(f"Files to touch (declared): {files}")
        total_read = 0
        for path_str in files:
            try:
                path = Path(path_str)
                candidate = (self.workspace / path)
                resolved = candidate.resolve()
                if not (
                    resolved == self.workspace_resolved
                    or self.workspace_resolved in resolved.parents
                ):
                    continue
                if not resolved.exists() or not resolved.is_file():
                    continue
                content = resolved.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if len(content) > MAX_FILE_CONTEXT:
                content = content[:MAX_FILE_CONTEXT] + (
                    f"\n... (truncated to {MAX_FILE_CONTEXT} chars)"
                )
            total_read += len(content)
            if total_read > MAX_TOTAL_CONTEXT:
                break
            try:
                rel = str(resolved.relative_to(self.workspace_resolved))
            except Exception:
                rel = str(path_str)
            parts.append(f"\n--- {rel} ---\n{content}\n--- end {rel} ---")
        try:
            listing = self.client.list_files(self.workspace_resolved)
            if listing:
                truncated = listing[:200]
                parts.append("\nWorkspace listing (top levels, capped):")
                for item in truncated:
                    parts.append(f"  {item}")
        except Exception:
            pass
        return "\n".join(parts)

    def _should_apply(self) -> bool:
        return bool(self.effective_settings.apply_worker_edits)

    async def run(self) -> str:
        context = self._build_context()
        prompt = f"{WORKER_PROMPT}\n\n{context}"
        try:
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        except Exception as exc:
            return f"Worker failed: {exc}"

        try:
            text = response if isinstance(response, str) else str(response)
        except Exception:
            text = ""
        model_summary = ""
        edits: list = []
        try:
            if text.strip():
                data = extract_json_object(text)
                model_summary = data.get("summary") or ""
                edits = data.get("edits") or []
            else:
                model_summary = ""
                edits = []
        except LLMParseError:
            model_summary = text.strip()
            edits = []
        except Exception:
            try:
                model_summary = text.strip()
            except Exception:
                model_summary = ""
            edits = []
        try:
            if text.strip():
                data = extract_json_object(text)
                model_summary = data.get("summary") or ""
                edits = data.get("edits") or []
            else:
                model_summary = ""
                edits = []
        except LLMParseError:
            model_summary = text.strip()
            edits = []
        except Exception:
            model_summary = text.strip() if hasattr(text, "strip") else str(text)
            edits = []

        apply = self._should_apply()
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            path_str = edit.get("path")
            content = edit.get("content")
            if not isinstance(path_str, str) or not path_str:
                self.rejected_edits.append(str(path_str) if path_str else "<missing>")
                continue
            if not isinstance(content, str):
                self.rejected_edits.append(path_str)
                continue
            if len(content) > MAX_EDIT_SIZE:
                self.rejected_edits.append(path_str)
                continue
            try:
                candidate = (self.workspace / path_str)
                resolved = candidate.resolve()
                if not (
                    resolved == self.workspace_resolved
                    or self.workspace_resolved in resolved.parents
                ):
                    self.rejected_edits.append(path_str)
                    continue
                if not apply:
                    self.rejected_edits.append(path_str)
                    continue
                if resolved.exists() and resolved.is_dir():
                    self.rejected_edits.append(path_str)
                    continue
                await self.client.write_file(resolved, content)
                try:
                    rel = str(resolved.relative_to(self.workspace_resolved))
                except Exception:
                    rel = str(path_str)
                if rel not in self.files_changed:
                    self.files_changed.append(rel)
            except Exception:
                self.rejected_edits.append(path_str)
                continue

        summary = f"Updated {len(self.files_changed)} file(s)"
        if self.files_changed:
            summary += ": " + ", ".join(self.files_changed)
        if model_summary:
            if self.files_changed or self.rejected_edits:
                summary += f" — {model_summary}"
            else:
                summary = model_summary
        return summary
