from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    pass


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        config: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient()
        self.config = config or client.settings if client else None

    async def run(self) -> str:
        # First, read any relevant files to understand context
        context = ""
        if self.task.files:
            for file_path in self.task.files:
                try:
                    content = await self.client.read_file(file_path)
                    context += f"\n--- {file_path} ---\n{content}\n"
                except Exception:
                    context += f"\n--- {file_path} ---\n(File not found or unreadable)\n"

        # Get list of files in workspace for reference
        workspace_files = self.client.list_files(self.config.workspace if self.config else Path.cwd())
        
        prompt = f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any relevant files'}
Dependencies: {', '.join(self.task.dependencies) if self.task.dependencies else 'none'}

Workspace files:
{', '.join(workspace_files[:50])}

Existing file context:
{context if context else '(no existing files to read)'}

Your response should be a JSON object with this shape:
{{
  "actions": [
    {{"type": "read", "path": "file.py"}},
    {{"type": "write", "path": "file.py", "content": "..."}},
    {{"type": "list", "path": "src/"}}
  ],
  "summary": "What you did and any issues"
}}

Only include actions you need to complete the task. Return JSON only."""

        response = await self.client.complete(
            prompt, 
            model=self.client.settings.worker_model
        )
        
        try:
            data = json.loads(response)
            actions = data.get("actions", [])
            summary = data.get("summary", "Task completed")
            
            for action in actions:
                await self._execute_action(action)
                
            return summary
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from response
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    actions = data.get("actions", [])
                    summary = data.get("summary", response)
                    for action in actions:
                        await self._execute_action(action)
                    return summary
                except json.JSONDecodeError:
                    pass
            return response

    async def _execute_action(self, action: dict) -> None:
        action_type = action.get("type")
        path = action.get("path")
        
        if action_type == "read":
            try:
                content = await self.client.read_file(path)
                return content
            except Exception as e:
                raise Exception(f"Failed to read {path}: {e}")
        elif action_type == "write":
            content = action.get("content", "")
            try:
                await self.client.write_file(path, content)
            except Exception as e:
                raise Exception(f"Failed to write {path}: {e}")
        elif action_type == "list":
            try:
                return self.client.list_files(path)
            except Exception as e:
                raise Exception(f"Failed to list {path}: {e}")
        else:
            raise Exception(f"Unknown action type: {action_type}")
