from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.agents.tools import WORKER_TOOLS, execute_tool
from furrow.config import Provider, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        if self.client.settings.provider == Provider.ANTHROPIC:
            return await self._run_anthropic(prompt)
        if self.client.settings.provider == Provider.OPENAI:
            return await self._run_openai(prompt)
        return await self.client.complete(prompt, model=self.client.settings.worker_model)

    async def _run_anthropic(self, prompt: str) -> str:
        max_rounds = max(1, getattr(self.client.settings, "worker_max_rounds", 8))
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        for _ in range(max_rounds):
            response = await self.client.anthropic.messages.create(
                model=self.client.settings.worker_model,
                max_tokens=4096,
                system=WORKER_PROMPT,
                tools=WORKER_TOOLS,
                messages=messages,
            )
            text = "".join(getattr(block, "text", "") for block in response.content)
            tool_uses = [block for block in response.content if getattr(block, "type", None) == "tool_use"]
            if not tool_uses:
                return text or "(no output)"
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for use in tool_uses:
                result = await execute_tool(use.name, dict(use.input))
                results.append({"type": "tool_result", "tool_use_id": use.id, "content": result})
            messages.append({"role": "user", "content": results})
        return "Worker reached the maximum number of rounds without a final answer."

    async def _run_openai(self, prompt: str) -> str:
        max_rounds = max(1, getattr(self.client.settings, "worker_max_rounds", 8))
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in WORKER_TOOLS
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": WORKER_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for _ in range(max_rounds):
            response = await self.client.openai.chat.completions.create(
                model=self.client.settings.worker_model,
                tools=tools,
                messages=messages,
            )
            message = response.choices[0].message
            if not message.tool_calls:
                return message.content or "(no output)"
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [tc.model_dump(exclude_none=True) for tc in message.tool_calls],
                }
            )
            for call in message.tool_calls:
                args: dict[str, Any] = {}
                if call.function.arguments:
                    try:
                        args = json.loads(call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                result = await execute_tool(call.function.name, args)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        return "Worker reached the maximum number of rounds without a final answer."