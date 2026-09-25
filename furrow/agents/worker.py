from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = self.client.settings.workspace
        self.max_rounds = 3
        self.round = 0

    async def run(self) -> str:
        # Read relevant files first
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(self.workspace / file_path)
                file_contents[file_path] = content
            except Exception:
                file_contents[file_path] = f"[File not found: {file_path}]"

        context = self._build_context(file_contents)
        
        for self.round in range(self.max_rounds):
            prompt = self._build_prompt(context)
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            
            # Parse response for tool calls
            action = self._parse_response(response)
            if action is None:
                # No more actions, task is done
                return response
            
            # Execute the action
            result = await self._execute_action(action)
            context += f"\n\nAction result:\n{result}"
        
        return f"Max rounds ({self.max_rounds}) reached. Last response: {response}"

    def _build_context(self, file_contents: dict[str, str]) -> str:
        ctx = f"Task: {self.task.description}\n"
        if self.task.files:
            ctx += f"Files to work on: {', '.join(self.task.files)}\n"
        if file_contents:
            ctx += "\n--- File Contents ---\n"
            for path, content in file_contents.items():
                ctx += f"\n### {path}\n```\n{content}\n```\n"
        return ctx

    def _build_prompt(self, context: str) -> str:
        return f"""{WORKER_PROMPT}

{context}

You can perform the following actions by responding with a JSON object (no markdown):
1. {{"action": "read_file", "path": "relative/path.py"}} - Read a file
2. {{"action": "write_file", "path": "relative/path.py", "content": "file content"}} - Write/create a file
3. {{"action": "edit_file", "path": "relative/path.py", "old": "old text", "new": "new text"}} - Edit a file
4. {{"action": "list_files", "directory": "relative/dir"}} - List files in directory
5. {{"action": "run_command", "command": ["cmd", "arg1", "arg2"]}} - Run a shell command
6. {{"action": "done", "summary": "Task completed. Summary of changes."}} - Mark task as complete

Respond with ONLY a JSON object. If the task is complete, use the "done" action with a summary."""

    def _parse_response(self, response: str) -> dict[str, Any] | None:
        try:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                action = json.loads(json_str)
                if action.get("action") == "done":
                    return None  # Signal completion
                return action
        except json.JSONDecodeError:
            pass
        return None

    async def _execute_action(self, action: dict[str, Any]) -> str:
        act = action.get("action")
        try:
            if act == "read_file":
                path = self.workspace / action["path"]
                content = await self.client.read_file(path)
                return f"File content:\n{content}"
            elif act == "write_file":
                path = self.workspace / action["path"]
                await self.client.write_file(path, action["content"])
                return f"File written: {action['path']}"
            elif act == "edit_file":
                path = self.workspace / action["path"]
                content = await self.client.read_file(path)
                old_text = action["old"]
                new_text = action["new"]
                if old_text in content:
                    new_content = content.replace(old_text, new_text, 1)
                    await self.client.write_file(path, new_content)
                    return f"File edited: {action['path']}"
                else:
                    return f"ERROR: Old text not found in {action['path']}"
            elif act == "list_files":
                directory = self.workspace / action.get("directory", ".")
                files = self.client.list_files(directory)
                return f"Files in {action.get('directory', '.')}:\n" + "\n".join(files)
            elif act == "run_command":
                import asyncio
                cmd = action["command"]
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=self.workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                return f"Exit code: {proc.returncode}\nstdout:\n{stdout.decode()}\nstderr:\n{stderr.decode()}"
            else:
                return f"ERROR: Unknown action: {act}"
        except Exception as e:
            return f"ERROR executing {act}: {str(e)}"
