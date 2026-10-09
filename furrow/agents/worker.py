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
        files_context = ""
        for f in self.task.files:
            try:
                content = await self.client.read_file(f)
                files_context += f"\n--- {f} ---\n{content}\n"
            except Exception:
                pass

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles: {', '.join(self.task.files) if self.task.files else 'any'}\n{files_context}"

        messages = [{"role": "user", "content": prompt}]

        for _ in range(5):
            response = await self.client.complete_with_messages(messages, model=self.client.settings.worker_model)

            try:
                tool_call = json.loads(response.strip())
                tool_name = tool_call.get("tool")
                if tool_name == "write":
                    path = tool_call["path"]
                    content = tool_call["content"]
                    await self.client.write_file(path, content)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Wrote {path}"})
                elif tool_name == "read":
                    path = tool_call["path"]
                    content = await self.client.read_file(path)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"File {path}:\n{content}"})
                else:
                    return response
            except (json.JSONDecodeError, KeyError):
                return response

        return "Max tool rounds reached"
