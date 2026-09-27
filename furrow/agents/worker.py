from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import FileOperation, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        file_contents: dict[str, str] = {}
        for path in self.task.files:
            try:
                content = await self.client.read_file(path)
            except FileNotFoundError:
                content = ""
            file_contents[path] = content

        context_lines = [f"{WORKER_PROMPT}", "", f"Task: {self.task.description}", ""]
        for path, content in file_contents.items():
            if content:
                context_lines.append(f"--- {path} (existing) ---")
                context_lines.append(content)
                context_lines.append("")
            else:
                context_lines.append(f"--- {path} (new file) ---")
                context_lines.append("")

        prompt = "\n".join(context_lines)

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        operations = self._parse_file_operations(response)
        modified: list[str] = []
        for op in operations:
            if op.operation == "write" and op.content is not None:
                await self.client.write_file(op.path, op.content)
                modified.append(op.path)
            elif op.operation == "edit" and op.old is not None and op.new is not None:
                await self._apply_edit(op.path, op.old, op.new)
                modified.append(op.path)

        if not modified:
            return response.strip()

        summary = f"Modified {len(modified)} file(s): {', '.join(modified)}"
        return f"{summary}\n\n{response.strip()}"

    @staticmethod
    def _parse_file_operations(response: str) -> list[FileOperation]:
        """Parse <write> and <edit> XML blocks from an LLM response.

        Results are returned in document order (the order they appear in the
        response text), regardless of tag type.
        """
        operations: list[FileOperation] = []

        write_pattern = r'<write\s+path="([^"]+)">(.*?)</write>'
        edit_pattern = r'<edit\s+path="([^"]+)">(.*?)</edit>'
        old_new_pattern = r'<old>(.*?)</old>\s*<new>(.*?)</new>'

        # Collect all matches with their byte offset so we can sort by position
        matches: list[tuple[int, str, re.Match[str]]] = []
        for m in re.finditer(write_pattern, response, re.DOTALL):
            matches.append((m.start(), "write", m))
        for m in re.finditer(edit_pattern, response, re.DOTALL):
            matches.append((m.start(), "edit", m))

        # Sort by position to preserve document order
        matches.sort(key=lambda item: item[0])

        for _, op_type, match in matches:
            if op_type == "write":
                path, content = match.group(1), match.group(2)
                operations.append(FileOperation(operation="write", path=path, content=content))
            else:
                path, inner = match.group(1), match.group(2)
                sub = re.search(old_new_pattern, inner, re.DOTALL)
                if sub:
                    operations.append(
                        FileOperation(
                            operation="edit",
                            path=path,
                            old=sub.group(1),
                            new=sub.group(2),
                        )
                    )

        return operations

    async def _apply_edit(self, path: str, old: str, new: str) -> None:
        content = await self.client.read_file(path)
        updated = content.replace(old, new, 1)
        await self.client.write_file(path, updated)
