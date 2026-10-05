from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT, WORKER_SYSTEM_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        workspace = Path(self.client.settings.workspace).resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        # Gather context: file listing and contents of relevant files
        file_listing = self.client.list_files(workspace)
        file_contents: dict[str, str] = {}

        for file_path in self.task.files:
            full_path = workspace / file_path
            if full_path.exists() and full_path.is_file():
                try:
                    content = await self.client.read_file(full_path)
                    # Truncate to reasonable size (e.g., 10000 chars)
                    if len(content) > 10000:
                        content = (
                            content[:10000] + f"\n... (+{len(content) - 10000} chars truncated)"
                        )
                    file_contents[file_path] = content
                except OSError as e:
                    log.warning("worker_read_file_failed", path=file_path, error=str(e))

        # Build prompt with context
        context_parts = [
            f"Task: {self.task.description}",
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}",
        ]

        if file_listing:
            context_parts.append(f"Workspace files:\n{chr(10).join(file_listing)}")

        if file_contents:
            context_parts.append("Current file contents:")
            for path, content in file_contents.items():
                context_parts.append(f"\n--- {path} ---\n{content}")

        prompt = f"{WORKER_PROMPT}\n\n" + "\n\n".join(context_parts)

        response = await self.client.complete(
            prompt, system=WORKER_SYSTEM_PROMPT, model=self.client.settings.worker_model
        )

        try:
            data = self.client.extract_json(response)
        except ValueError as e:
            log.warning("worker_json_parse_failed", error=str(e))
            return f"Worker failed to parse response as JSON: {e}"

        summary = str(data.get("summary", "")).strip()
        files_data = data.get("files", [])

        if not isinstance(files_data, list):
            files_data = []

        written_files: list[str] = []
        rejected_files: list[str] = []

        for file_entry in files_data:
            if not isinstance(file_entry, dict):
                continue
            path = str(file_entry.get("path", "")).strip()
            content = file_entry.get("content", "")
            if not path or not isinstance(content, str):
                continue

            # Path safety: resolve against workspace and ensure it doesn't escape
            try:
                target_path = (workspace / path).resolve()
                # Check if the resolved path is within workspace
                target_path.relative_to(workspace)
            except ValueError:
                rejected_files.append(path)
                log.warning("worker_path_escape_rejected", path=path, workspace=str(workspace))
                continue

            try:
                await self.client.write_file(target_path, content)
                written_files.append(path)
            except OSError as e:
                log.error("worker_write_file_failed", path=path, error=str(e))

        result_parts = []
        if written_files:
            result_parts.append(f"Written: {', '.join(written_files)}")
        if rejected_files:
            result_parts.append(f"Rejected (path escape): {', '.join(rejected_files)}")
        if not written_files and not rejected_files:
            result_parts.append("No files written")

        if summary:
            return f"{' | '.join(result_parts)} | {summary}"
        return " | ".join(result_parts)
