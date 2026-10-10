from __future__ import annotations

import json
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
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            operations = data.get("operations", [])
            summary = data.get("summary", "No summary provided")
        except (json.JSONDecodeError, ValueError) as e:
            return f"Failed to parse worker response: {e}\nResponse: {response}"

        results = []
        for op in operations:
            op_type = op.get("type")
            try:
                if op_type == "read":
                    content = await self.client.read_file(op["path"])
                    results.append(f"Read {op['path']} ({len(content)} chars)")
                elif op_type == "write":
                    await self.client.write_file(op["path"], op["content"])
                    results.append(f"Wrote {op['path']}")
                elif op_type == "edit":
                    content = await self.client.read_file(op["path"])
                    old_text = op["old"]
                    new_text = op["new"]
                    if old_text not in content:
                        results.append(f"Edit failed: text not found in {op['path']}")
                    else:
                        new_content = content.replace(old_text, new_text, 1)
                        await self.client.write_file(op["path"], new_content)
                        results.append(f"Edited {op['path']}")
                else:
                    results.append(f"Unknown operation type: {op_type}")
            except (OSError, KeyError, ValueError) as e:
                results.append(f"Operation {op_type} failed: {e}")

        return f"{summary}\n" + "\n".join(results)
