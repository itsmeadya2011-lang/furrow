from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import structlog

from furrow.llm import LLMClient

logger = structlog.get_logger(__name__)


@dataclass
class FileSummary:
    path: str
    summary: str
    last_modified_cycle: int
    lines_changed: int = 0


class ContextManager:
    """Manages project context for the Furrow loop."""

    def __init__(self, workspace: Path, client: LLMClient):
        self.workspace = workspace
        self.client = client
        self.file_summaries: dict[str, FileSummary] = {}
        self.recent_changes: list[dict] = []
        self.cycle_summaries: list[str] = []
        self.max_recent_changes = 20
        self.max_cycle_summaries = 10

    async def update_from_changes(self, changes: list[dict], cycle: int) -> None:
        """Update context from file changes."""
        for change in changes:
            file_path = change.get("file", "")
            if not file_path:
                continue

            content = change.get("content", "")
            summary = await self._summarize_file(file_path, content)

            self.file_summaries[file_path] = FileSummary(
                path=file_path,
                summary=summary,
                last_modified_cycle=cycle,
                lines_changed=len(content.splitlines()),
            )

            self.recent_changes.append(
                {
                    "cycle": cycle,
                    "file": file_path,
                    "summary": summary,
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )

        # Trim old entries
        if len(self.recent_changes) > self.max_recent_changes:
            self.recent_changes = self.recent_changes[-self.max_recent_changes :]

    async def _summarize_file(self, path: str, content: str) -> str:
        """Generate a summary of a file using the LLM."""
        if not content.strip():
            return "Empty file"

        prompt = f"""Summarize this file in 1-2 sentences. Focus on what it does, key functions/classes, and its purpose.

File: {path}
Content:
{content[:3000]}"""

        try:
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            return response.strip()[:200]
        except Exception:  # noqa: BLE001 - best-effort fallback for any LLM failure
            # Fallback to simple summary
            lines = content.splitlines()
            return f"{len(lines)} lines, {len(content)} chars"

    def add_cycle_summary(self, summary: str) -> None:
        """Add a summary of what was accomplished in a cycle."""
        self.cycle_summaries.append(summary)
        if len(self.cycle_summaries) > self.max_cycle_summaries:
            self.cycle_summaries = self.cycle_summaries[-self.max_cycle_summaries :]

    def get_context_for_planner(self) -> str:
        """Get context string for the planner."""
        parts = []

        if self.file_summaries:
            parts.append("## Project Files (recent summaries)")
            for path, summary in sorted(self.file_summaries.items()):
                parts.append(f"- `{path}`: {summary.summary}")

        if self.recent_changes:
            parts.append("\n## Recent Changes")
            for change in self.recent_changes[-5:]:
                parts.append(f"- Cycle {change['cycle']}: {change['file']} - {change['summary']}")

        if self.cycle_summaries:
            parts.append("\n## Recent Cycles")
            for i, summary in enumerate(self.cycle_summaries[-3:], 1):
                parts.append(f"- Cycle {len(self.cycle_summaries) - 3 + i}: {summary}")

        return "\n".join(parts) if parts else "No context available."

    def get_context_for_worker(self, task_files: list[str]) -> str:
        """Get relevant context for a specific task."""
        parts = []

        relevant_files = [f for f in task_files if f in self.file_summaries]
        if relevant_files:
            parts.append("## Relevant File Summaries")
            for path in relevant_files:
                summary = self.file_summaries[path]
                parts.append(f"- `{path}`: {summary.summary}")

        # Also include recently changed files that might be related
        recent_files = [
            c["file"] for c in self.recent_changes[-5:] if c["file"] not in relevant_files
        ]
        if recent_files:
            parts.append("\n## Recently Modified Files")
            for path in recent_files:
                if path in self.file_summaries:
                    summary = self.file_summaries[path]
                    parts.append(f"- `{path}`: {summary.summary}")

        return "\n".join(parts) if parts else "No specific context."
