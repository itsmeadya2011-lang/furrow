from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

# Extended prompt that instructs the LLM to output file changes in JSON format
WORKER_PROMPT_WITH_WRITES = WORKER_PROMPT + """

IMPORTANT: After implementing the task, output a JSON block with the files you modified:
```json
{
  "files": [
    {
      "path": "src/file.py",
      "action": "create" | "modify" | "delete",
      "content": "full file content here (for create/modify)"
    }
  ]
}
```

If you don't need to create or modify any files, return:
```json
{"files": []}
```
"""


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT_WITH_WRITES}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Try to extract and apply file changes
        await self._apply_file_changes(response)
        
        return self._summarize_changes(response)

    async def _apply_file_changes(self, response: str) -> None:
        """Extract JSON file changes from LLM response and write them to disk."""
        # Look for JSON block in the response
        json_match = re.search(r"```json\s*({\s*\"files\"\s*:.*?})\s*```", response, re.DOTALL)
        if not json_match:
            # Try without code block markers
            json_match = re.search(r"({\s*\"files\"\s*:\s*\[.*?\]\s*})", response, re.DOTALL)
        
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                files = data.get("files", [])
                workspace = self.client.settings.workspace
                
                for file_info in files:
                    path = file_info.get("path", "")
                    action = file_info.get("action", "modify")
                    content = file_info.get("content", "")
                    
                    if not path:
                        continue
                    
                    file_path = workspace / path
                    
                    if action == "delete":
                        if file_path.exists():
                            file_path.unlink()
                    elif action in ("create", "modify"):
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        await self.client.write_file(file_path, content)
            except (json.JSONDecodeError, KeyError) as e:
                # If parsing fails, silently ignore - the summary will still be returned
                pass

    def _summarize_changes(self, response: str) -> str:
        """Extract a summary from the LLM response, removing the JSON block."""
        # Remove JSON blocks for cleaner summary
        cleaned = re.sub(r"```json\s*\{.*?\}\s*```", "", response, flags=re.DOTALL).strip()
        return cleaned if cleaned else response.strip()
