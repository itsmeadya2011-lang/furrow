from __future__ import annotations

import json
import re
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
        # Read existing files to give the LLM context
        context_parts: list[str] = []
        for f in self.task.files:
            try:
                content = await self.client.read_file(f)
                context_parts.append(f"--- {f} (existing) ---\n{content}\n")
            except FileNotFoundError:
                context_parts.append(f"--- {f} (new file) ---\n")

        files_hint = ", ".join(self.task.files) if self.task.files else "any relevant files"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_hint}\n\n"
            + "\n".join(context_parts)
            + "\n\nReturn JSON only (no markdown, no explanation) with this shape:\n"
            '{"files": [{"path": "src/example.py", "content": "..."}], "summary": "..."}'
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Try to extract JSON from the response
        data = self._parse_response(response)
        if data is None:
            # Fallback: try to find file blocks in markdown code fences
            data = self._parse_markdown_files(response)

        written: list[str] = []
        if data and "files" in data:
            for f in data["files"]:
                path = f.get("path", "")
                content = f.get("content", "")
                if path and content:
                    await self.client.write_file(path, content)
                    written.append(path)

        summary = data.get("summary", response[:200]) if data else response[:200]
        if written:
            return f"Wrote {len(written)} file(s): {', '.join(written)}. {summary}"
        return f"No files written. {summary}"

    def _parse_response(self, response: str) -> dict | None:
        # Try direct JSON parse
        try:
            return json.loads(response)
        except (json.JSONDecodeError, ValueError):
            pass
        # Try to extract JSON from code fences or surrounding text
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    def _parse_markdown_files(self, response: str) -> dict | None:
        # Fallback: parse ```language\npath\n```\ncontent blocks
        pattern = re.compile(
            r"```(?:\w+)?\s*\n?([^\n`]+)\n```(.*?)(?=```|\Z)", re.DOTALL
        )
        matches = pattern.findall(response)
        if not matches:
            return None
        files: list[dict] = []
        for path, content in matches:
            path = path.strip()
            if path and not path.startswith("<"):
                files.append({"path": path, "content": content.strip()})
        if files:
            return {"files": files, "summary": "Parsed from markdown"}
        return None