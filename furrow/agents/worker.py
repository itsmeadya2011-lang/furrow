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
        # Read context from files if specified
        context_parts = []
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                context_parts.append(f"=== {file_path} ===\n{content}\n")
            except Exception as e:
                context_parts.append(f"=== {file_path} ===\n[Error reading file: {e}]\n")

        file_context = "\n".join(context_parts) if context_parts else "No specific files provided - explore as needed."

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n\nFile context:\n{file_context}\n\nReturn a JSON response with changes to make. Include files you modified with their full new content."
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Try to extract and apply file changes
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                changes = data.get('changes', data.get('files', []))
                for change in changes:
                    file_path = change.get('file') or change.get('path')
                    content = change.get('content') or change.get('new_content')
                    if file_path and content is not None:
                        await self.client.write_file(file_path, content)
                return data.get('summary', response)
            # Look for code blocks that specify files
            # Format: ```filename\ncontent```
            file_blocks = re.findall(r'```(?:\S+)?\s*(.*?)```', response, re.DOTALL)
            # If no explicit file markers, just return summary
            if file_blocks and self.task.files:
                # Try to apply to first specified file if content looks like replacement
                return response
        except Exception as e:
            pass  # Fall through to returning response

        return response
