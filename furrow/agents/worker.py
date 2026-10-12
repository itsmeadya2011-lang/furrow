from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings as SettingsType

logger = logging.getLogger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings_obj: SettingsType | None = None) -> None:
        self.task = task
        effective_settings = settings_obj or settings
        self.client = client or LLMClient(settings=effective_settings)
        self.workspace = effective_settings.workspace

    async def run(self) -> str:
        # Read relevant files first to provide context
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.workspace / file_path
            if full_path.exists():
                try:
                    content = await self.client.read_file(full_path)
                    file_contents[file_path] = content
                except OSError as e:
                    logger.warning("file_read_failed: file=%s error=%s", file_path, e)
                    file_contents[file_path] = f"[Error reading file: {e}]"

        # Build context from file contents
        context_parts = []
        if file_contents:
            context_parts.append("Current file contents:")
            for path, content in file_contents.items():
                context_parts.append(f"\n--- {path} ---\n{content[:5000]}")

        context = "\n".join(context_parts) if context_parts else "No existing files to reference."

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n{context}\n\nYou can perform file operations by responding with JSON actions. Available actions:\n- {{\"action\": \"read\", \"file\": \"path/to/file\"}}\n- {{\"action\": \"write\", \"file\": \"path/to/file\", \"content\": \"...\"}}\n- {{\"action\": \"run\", \"cmd\": [\"cmd\", \"arg1\", \"arg2\"]}}\n- {{\"action\": \"done\", \"summary\": \"Task completed: ...\"}}\n\nRespond with a JSON array of actions to execute sequentially. The last action must be 'done' with a summary."

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        return await self._execute_actions(response)

    async def _execute_actions(self, response: str) -> str:
        """Parse and execute actions from LLM response."""
        # Try to extract JSON from response
        actions = self._parse_actions(response)
        if not actions:
            return f"Worker response (no actions parsed): {response[:500]}"

        results = []
        for action in actions:
            action_type = action.get("action")
            try:
                if action_type == "read":
                    file_path = self.workspace / action["file"]
                    content = await self.client.read_file(file_path)
                    results.append(f"Read {action['file']}: {len(content)} chars")
                elif action_type == "write":
                    file_path = self.workspace / action["file"]
                    await self.client.write_file(file_path, action["content"])
                    results.append(f"Wrote {action['file']}: {len(action['content'])} chars")
                elif action_type == "run":
                    exit_code, stdout, stderr = await self.client.run_command(action["cmd"])
                    results.append(f"Ran {' '.join(action['cmd'])}: exit={exit_code}")
                    if stdout:
                        results.append(f"  stdout: {stdout[:500]}")
                    if stderr:
                        results.append(f"  stderr: {stderr[:500]}")
                elif action_type == "done":
                    return action.get("summary", "Task completed")
                else:
                    results.append(f"Unknown action: {action_type}")
            except (OSError, KeyError, ValueError) as e:
                logger.warning("action_execution_failed: action=%s error=%s", action_type, e)
                results.append(f"Error executing {action_type}: {e}")

        return "\n".join(results)

    def _parse_actions(self, response: str) -> list[dict]:
        """Extract JSON actions from LLM response."""
        # Try to find JSON array in response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try to find JSON object (single action)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                obj = json.loads(json_match.group())
                return [obj]
            except json.JSONDecodeError:
                pass

        return []
