from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


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
        workspace = Path(self.client.settings.workspace)

        files_context = await self._gather_files(workspace)

        files_desc = (
            ", ".join(self.task.files) if self.task.files else "any"
        )

        prompt = (
            f"{WORKER_PROMPT}\n"
            f"\nTask: {self.task.description}\n"
            f"Files to touch: {files_desc}\n"
            f"\n=== Relevant file contents ===\n{files_context}\n"
            f"=== End file contents ===\n"
            f"\nRespond with the JSON object described above.\n"
        )

        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )

        data = self._parse_response(response)
        if data is None:
            return (
                "Failed to parse LLM response as JSON. Response:\n"
                f"{response[:1000]}"
            )

        changes = data.get("changes", [])
        summary = data.get("summary", "")

        written: list[str] = []
        for change in changes:
            rel_path = change.get("path", "")
            content = change.get("content", "")
            if not rel_path:
                continue
            abs_path = workspace / rel_path
            await self.client.write_file(abs_path, content)
            written.append(rel_path)

        result = f"Summary: {summary}\n" if summary else ""
        result += (
            f"Files written ({len(written)}): "
            f"{', '.join(written) if written else 'none'}"
        )
        return result

    async def _gather_files(self, workspace: Path) -> str:
        """Read the relevant files from the workspace for inclusion in the prompt."""
        if not self.task.files:
            listing = self.client.list_files(workspace)
            if not listing:
                return "<workspace is empty>"
            preview = ", ".join(listing[:50])
            return (
                "No specific files were assigned. The workspace contains "
                f"{len(listing)} file(s) (showing first 50): {preview}\n"
            )

        blocks: list[str] = []
        for rel_path in self.task.files:
            abs_path = workspace / rel_path
            try:
                content = await self.client.read_file(abs_path)
            except FileNotFoundError:
                blocks.append(
                    f"--- File: {rel_path} (does not exist yet; create it) ---"
                )
                continue
            except IsADirectoryError:
                rel_files = self.client.list_files(abs_path)
                listing = (
                    ", ".join(rel_files) if rel_files else "<empty directory>"
                )
                blocks.append(
                    f"--- File: {rel_path} is a directory; listing contents ---\n"
                    f"{listing}"
                )
                continue
            blocks.append(f"--- File: {rel_path} ---\n{content}")

        return "\n".join(blocks)

    @staticmethod
    def _parse_response(response: str) -> dict | None:
        """Parse the LLM response into a dict, tolerating markdown fences."""
        text = response.strip()

        # Strip leading ```[lang] fence and trailing ``` fence if present.
        text = re.sub(r"^```\w*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fall back to extracting the first JSON object in the response.
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
            return None
