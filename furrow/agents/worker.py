from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.agents.tools import TOOLS, execute_tool, openai_tools
from furrow.config import Provider, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

MAX_TOOL_ROUNDS = 12
MAX_TOKENS = 4096


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self.task = task
        self.settings = settings
        self.client = client or LLMClient(settings=settings)
        self.workspace = workspace or self.client.settings.workspace
        self.max_tool_rounds = max_tool_rounds

    async def run(self) -> str:
        model = self.client.settings.worker_model
        try:
            existing = self._existing_files()
        except Exception:
            existing = []
        files = "\n".join(existing) if existing else "(none)"
        user_msg = (
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Workspace: {self.workspace}\n"
            f"Existing files:\n{files}\n"
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]

        if self.client.settings.provider == Provider.ANTHROPIC:
            return await self._run_anthropic(model, messages)
        elif self.client.settings.provider == Provider.OPENAI:
            return await self._run_openai(model, messages)
        else:
            raise ValueError(f"Unsupported provider: {self.client.settings.provider}")

    def _existing_files(self) -> list[str]:
        return self.client.list_files(self.workspace)

    async def _run_anthropic(self, model: str, messages: list[dict[str, Any]]) -> str:
        tools = TOOLS
        for _ in range(self.max_tool_rounds):
            response = await self.client.anthropic.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=WORKER_PROMPT,
                tools=tools,
                messages=messages,
            )
            if response.stop_reason == "tool_use":
                assistant_blocks: list[dict[str, Any]] = []
                tool_results: list[dict[str, Any]] = []
                for block in response.content:
                    if block.type == "text":
                        assistant_blocks.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_blocks.append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            }
                        )
                        result = await execute_tool(block.name, block.input, self.workspace)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(result),
                            }
                        )
                messages.append({"role": "assistant", "content": assistant_blocks})
                messages.append({"role": "user", "content": tool_results})
            else:
                text_parts = [block.text for block in response.content if block.type == "text"]
                return "\n".join(text_parts) if text_parts else ""
        return "Reached maximum tool-use rounds without producing a final response."

    async def _run_openai(self, model: str, messages: list[dict[str, Any]]) -> str:
        messages.insert(0, {"role": "system", "content": WORKER_PROMPT})
        for _ in range(self.max_tool_rounds):
            response = await self.client.openai.chat.completions.create(
                model=model,
                max_tokens=MAX_TOKENS,
                messages=messages,
                tools=openai_tools(),
                tool_choice="auto",
            )
            msg = response.choices[0].message
            tool_calls = msg.tool_calls or []
            if tool_calls:
                assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
                messages.append(assistant_msg)
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = await execute_tool(tc.function.name, args, self.workspace)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "content": json.dumps(result),
                        }
                    )
            else:
                return msg.content or ""
        return "Reached maximum tool-use rounds without producing a final response."
