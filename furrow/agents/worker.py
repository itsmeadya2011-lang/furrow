from __future__ import annotations

import asyncio
import json

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or Settings())
        self.settings = settings or self.client.settings
        self.workspace = self.settings.workspace

    async def run(self) -> str:
        # Read relevant files first
        file_contents: dict[str, str] = {}
        for file_path in self.task.files:
            full_path = self.workspace / file_path
            if full_path.exists():
                file_contents[file_path] = await self.client.read_file(full_path)

        # Build prompt with context
        files_context = ""
        if file_contents:
            files_context = "\n\nRelevant files:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # The worker should return a JSON with actions to perform
        # For now, we'll parse the response and try to extract file changes
        return await self._execute_response(response)

    async def _execute_response(self, response: str) -> str:
        """Parse the LLM response and execute any file operations or commands."""
        # Try to parse as JSON for structured actions
        try:
            actions = json.loads(response)
            if isinstance(actions, dict) and "actions" in actions:
                return await self._execute_actions(actions["actions"])
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: treat as a summary
        return response

    async def _execute_actions(self, actions: list[dict]) -> str:
        """Execute a list of actions (file writes, commands, etc.)."""
        results = []
        
        for action in actions:
            action_type = action.get("type")
            
            if action_type == "write_file":
                path = self.workspace / action["path"]
                content = action["content"]
                await self.client.write_file(path, content)
                results.append(f"Wrote {action['path']}")
                
            elif action_type == "read_file":
                path = self.workspace / action["path"]
                content = await self.client.read_file(path)
                results.append(f"Read {action['path']}: {content[:200]}...")
                
            elif action_type == "run_command":
                cmd = action["command"]
                cwd = self.workspace
                if "cwd" in action:
                    cwd = self.workspace / action["cwd"]
                
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                output = stdout.decode() + stderr.decode()
                results.append(f"Command {' '.join(cmd)}: {output[:500]}")
                
            elif action_type == "list_files":
                directory = self.workspace
                if "path" in action:
                    directory = self.workspace / action["path"]
                files = self.client.list_files(directory)
                results.append(f"Files in {action.get('path', '.')}: {files}")
                
            else:
                results.append(f"Unknown action type: {action_type}")
        
        return "\n".join(results)