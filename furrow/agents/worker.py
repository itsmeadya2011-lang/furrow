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
        self.max_rounds = 3

    async def run(self) -> str:
        """Run the worker agent to implement the task."""
        # First, read any relevant files to understand the codebase
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = f"# File not found: {file_path}\n"
            except Exception as e:
                file_contents[file_path] = f"# Error reading {file_path}: {e}\n"

        # Build context from existing files
        context = ""
        if file_contents:
            context = "\n\nExisting files:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{context}\n\nRespond with a JSON object containing your changes:\n{{\n  \"changes\": [\n    {{\n      \"file\": \"path/to/file.py\",\n      \"content\": \"full file content here\"\n    }}\n  ],\n  \"summary\": \"Brief description of changes made\"\n}}"

        for round_num in range(self.max_rounds):
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            
            try:
                data = json.loads(response)
                changes = data.get("changes", [])
                summary = data.get("summary", "Changes applied")
                
                # Apply changes
                for change in changes:
                    file_path = change.get("file")
                    content = change.get("content", "")
                    if file_path:
                        await self.client.write_file(file_path, content)
                
                if not changes:
                    return f"No changes needed. {summary}"
                
                return f"Applied {len(changes)} file change(s). {summary}"
                
            except (json.JSONDecodeError, ValueError) as e:
                # If not valid JSON, try to extract code blocks
                changes = self._extract_changes_from_response(response)
                if changes:
                    for change in changes:
                        await self.client.write_file(change["file"], change["content"])
                    return f"Applied {len(changes)} file change(s) (extracted from response)"
                
                # If last round, return the response as-is
                if round_num == self.max_rounds - 1:
                    return f"Worker completed (text response): {response[:500]}"
                
                # Otherwise, ask for JSON format
                prompt = f"Please respond with valid JSON only. Your previous response was not valid JSON: {e}\n\n{prompt}"

        return "Worker completed without valid output"

    def _extract_changes_from_response(self, response: str) -> list[dict[str, str]]:
        """Extract file changes from markdown code blocks in the response."""
        changes = []
        
        # Pattern to match ```python or ```filepath\ncontent\n```
        pattern = r"```(?:(\w+)\n)?(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)
        
        for lang_or_path, content in matches:
            # Try to determine file path from language or content
            file_path = None
            
            # Check if first line looks like a file path
            lines = content.strip().split('\n')
            if lines and ('/' in lines[0] or '\\' in lines[0] or lines[0].endswith(('.py', '.js', '.ts', '.json', '.md', '.txt', '.yaml', '.yml'))):
                file_path = lines[0].strip()
                content = '\n'.join(lines[1:])
            elif lang_or_path and lang_or_path not in ['python', 'javascript', 'typescript', 'json', 'yaml', 'markdown', 'bash', 'sh']:
                file_path = lang_or_path
            
            if file_path:
                changes.append({"file": file_path, "content": content})
        
        return changes