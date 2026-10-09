from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_rounds = 3
        self.rounds = 0

    async def run(self) -> str:
        """Execute the task by iterating: read files, plan changes, implement, verify."""
        workspace = self.client.settings.workspace
        
        # First, read relevant files to understand the codebase
        files_content = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(workspace / file_path)
                files_content[file_path] = content
            except FileNotFoundError:
                files_content[file_path] = f"[FILE NOT FOUND: {file_path}]"
        
        # Build initial prompt with file contents
        files_section = "\n\n".join(
            f"=== {path} ===\n{content}" 
            for path, content in files_content.items()
        ) if files_content else "No existing files specified. Create new files as needed."
        
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n\nFiles to work on:\n{files_section}\n"
        
        # Iterative implementation loop
        for round_num in range(self.max_rounds):
            self.rounds = round_num + 1
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            
            # Try to parse structured response with file operations
            try:
                data = json.loads(response)
                if isinstance(data, dict) and "files" in data:
                    # Apply file changes
                    changes_made = await self._apply_file_changes(data["files"])
                    if changes_made:
                        # Re-read modified files for next round
                        for file_path in data["files"]:
                            try:
                                content = await self.client.read_file(workspace / file_path)
                                files_content[file_path] = content
                            except FileNotFoundError:
                                pass
                        
                        files_section = "\n\n".join(
                            f"=== {path} ===\n{content}" 
                            for path, content in files_content.items()
                        )
                        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n\nFiles updated. Current state:\n{files_section}\n\nContinue if needed, or return final summary."
                        continue
            except json.JSONDecodeError:
                pass
            
            # If not structured file ops, treat as final summary
            return response
        
        return f"Completed {self.max_rounds} rounds. Final state: {self.task.description}"

    async def _apply_file_changes(self, files: dict[str, str]) -> bool:
        """Apply file changes from LLM response. Returns True if any changes made."""
        workspace = self.client.settings.workspace
        changes_made = False
        
        for file_path, content in files.items():
            full_path = workspace / file_path
            if content == "[DELETE]":
                if full_path.exists():
                    full_path.unlink()
                    changes_made = True
            else:
                await self.client.write_file(full_path, content)
                changes_made = True
        
        return changes_made
