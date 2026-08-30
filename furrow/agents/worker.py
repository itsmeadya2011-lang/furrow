from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        workspace = Path(self.client.settings.workspace)

        file_contexts: dict[str, str | None] = {}
        for rel_path in self.task.files:
            full_path = workspace / rel_path
            if full_path.exists():
                try:
                    file_contexts[rel_path] = await self.client.read_file(full_path)
                except Exception:
                    file_contexts[rel_path] = None
            else:
                file_contexts[rel_path] = None

        workspace_files = self.client.list_files(workspace)

        prompt_parts = [
            f"Task: {self.task.description}",
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}",
            "",
            "Existing files in workspace:",
            "\n".join(f"  {f}" for f in workspace_files[:200]) if workspace_files else "  (empty)",
            "",
        ]

        if file_contexts:
            prompt_parts.append("Current file contents:")
            for rel_path, content in file_contexts.items():
                if content is not None:
                    prompt_parts.append(f"\n### {rel_path}\n```\n{content}\n```")
                else:
                    prompt_parts.append(f"\n### {rel_path} (file does not exist, create it)")
            prompt_parts.append("")

        prompt_parts.append(
            "Respond with the complete contents of each file you want to modify or create, "
            "using this exact format:\n"
            "\n"
            "## Files to modify\n"
            "\n"
            "### path/to/file.py\n"
            "```python\n"
            "# complete file content here\n"
            "```\n"
            "\n"
            "### another/file.py\n"
            "```python\n"
            "# complete file content here\n"
            "```\n"
            "\n"
            "## Summary\n"
            "Brief description of changes made"
        )

        prompt = WORKER_PROMPT + "\n\n" + "\n".join(prompt_parts)
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        modified = self._parse_files(response)

        for rel_path, content in modified.items():
            full_path = workspace / rel_path
            await self.client.write_file(full_path, content)

        return self._extract_summary(response, modified)

    def _parse_files(self, response: str) -> dict[str, str]:
        files: dict[str, str] = {}
        match = re.search(r"## Files to modify\n(.*?)(?=\n## Summary|\Z)", response, re.DOTALL)
        if not match:
            return files

        section = match.group(1)
        headers = list(re.finditer(r"### (.+?)\n", section))

        for i, header in enumerate(headers):
            path = header.group(1).strip()
            start = header.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(section)
            block_text = section[start:end]

            code_match = re.search(r"```(?:\w+)?\n(.*?)```", block_text, re.DOTALL)
            if code_match:
                files[path] = code_match.group(1)

        return files

    def _extract_summary(self, response: str, modified: dict[str, str]) -> str:
        m = re.search(r"## Summary\n(.*?)(?:\Z)", response, re.DOTALL)
        if m:
            return m.group(1).strip()

        if modified:
            return f"Modified {len(modified)} file(s): {', '.join(modified.keys())}"
        return "No files were modified."
