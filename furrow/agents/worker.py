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
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # 1. Read the files listed in the task to understand current state
        file_contents: dict[str, str] = {}
        for filepath in self.task.files:
            try:
                file_contents[filepath] = await self.client.read_file(filepath)
            except (FileNotFoundError, OSError):
                file_contents[filepath] = ""  # file doesn't exist yet

        # 2. Build the prompt with file contents
        files_section = ""
        for path, content in file_contents.items():
            files_section += f"\n--- {path} ---\n{content}\n--- end {path} ---\n"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        if files_section:
            prompt += f"\nCurrent file contents:{files_section}"

        # 3. Ask the LLM to generate the implementation
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # 4. Parse the LLM response to extract file changes
        file_changes = self._parse_file_changes(response)

        # 5. Actually write the changed files
        written_files: list[str] = []
        for change in file_changes:
            try:
                await self.client.write_file(change["path"], change["content"])
                written_files.append(change["path"])
            except OSError as e:
                return f"Failed to write {change['path']}: {e}"

        # 6. Return a concise summary
        if written_files:
            return f"Modified {len(written_files)} file(s): {', '.join(written_files)}"
        return f"No file changes made. Response: {response[:200]}"

    def _parse_file_changes(self, response: str) -> list[dict[str, str]]:
        """Parse the LLM response to extract file changes.

        Expects JSON like:
        {
          "files": [
            {"path": "src/example.py", "content": "def hello():\\n    return 'world'\\n"}
          ],
          "summary": "..."
        }
        """
        # Try to extract JSON from the response
        # First, try direct JSON parse
        try:
            data = json.loads(response)
            return self._extract_files_from_data(data)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON in code fences
        fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if fenced:
            try:
                data = json.loads(fenced.group(1))
                return self._extract_files_from_data(data)
            except (json.JSONDecodeError, TypeError):
                pass

        # Try to find a JSON object anywhere in the response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return self._extract_files_from_data(data)
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: treat the whole response as a single file content if a path is mentioned
        path_match = re.search(r'(?:path|file|filename)[:\s]*["\']?([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)', response, re.IGNORECASE)
        if path_match:
            return [{"path": path_match.group(1), "content": response}]

        return []

    def _extract_files_from_data(self, data: dict[str, any]) -> list[dict[str, str]]:
        """Extract file changes from a parsed JSON data dict."""
        if isinstance(data, dict) and "files" in data:
            files = data["files"]
            if isinstance(files, list):
                result = []
                for f in files:
                    if isinstance(f, dict) and "path" in f and "content" in f:
                        result.append({"path": str(f["path"]), "content": str(f["content"])})
                return result
        return []