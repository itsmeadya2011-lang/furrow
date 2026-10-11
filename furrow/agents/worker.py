from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT, WORKER_TOOLS
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_rounds = 10

    async def run(self) -> str:
        logger.info("worker_started", task_id=self.task.id, description=self.task.description[:100])
        system_prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        
        messages: list[dict] = [{"role": "user", "content": "Begin working on the task. Use the available tools to read, write, and list files as needed."}]
        
        for round_num in range(self.max_rounds):
            response = await self._call_llm_with_tools(system_prompt, messages)
            
            if response.get("tool_calls"):
                for tool_call in response["tool_calls"]:
                    result = await self._execute_tool(tool_call)
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
                    messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})
            else:
                # No tool calls, task is complete
                logger.info("worker_completed", task_id=self.task.id, result=response.get("content", "")[:200])
                return response.get("content", "Task completed (no output)")
        
        logger.warning("worker_max_rounds_reached", task_id=self.task.id)
        return "Max rounds reached. Task may be incomplete."

    async def _call_llm_with_tools(self, system: str, messages: list[dict]) -> dict:
        model = self.client.settings.worker_model
        
        if self.client.settings.provider == "anthropic":
            return await self._call_anthropic_with_tools(system, messages, model)
        elif self.client.settings.provider == "openai":
            return await self._call_openai_with_tools(system, messages, model)
        else:
            raise ValueError(f"Unsupported provider: {self.client.settings.provider}")

    async def _call_anthropic_with_tools(self, system: str, messages: list[dict], model: str) -> dict:
        # Convert messages to anthropic format
        anthro_messages = []
        for msg in messages:
            if msg["role"] == "tool":
                anthro_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": msg["tool_call_id"], "content": msg["content"]}
                    ]
                })
            elif msg.get("tool_calls"):
                anthro_messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"])
                        }
                        for tc in msg["tool_calls"]
                    ]
                })
            else:
                anthro_messages.append({"role": msg["role"], "content": msg["content"]})
        
        response = await self.client.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=anthro_messages,
            tools=WORKER_TOOLS,
        )
        
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })
        
        return {"content": content, "tool_calls": tool_calls}

    async def _call_openai_with_tools(self, system: str, messages: list[dict], model: str) -> dict:
        openai_messages = [{"role": "system", "content": system}]
        openai_messages.extend(messages)
        
        response = await self.client.openai.chat.completions.create(
            model=model,
            messages=openai_messages,
            tools=WORKER_TOOLS,
            tool_choice="auto",
        )
        
        msg = response.choices[0].message
        result = {"content": msg.content or ""}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in msg.tool_calls
            ]
        return result

    async def _execute_tool(self, tool_call: dict) -> str:
        name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
        
        logger.debug("tool_called", tool=name, args=args)
        
        try:
            if name == "read_file":
                result = await self.client.read_file(args["path"])
                logger.debug("tool_result", tool=name, result_length=len(result))
                return result
            elif name == "write_file":
                await self.client.write_file(args["path"], args["content"])
                return f"Successfully wrote to {args['path']}"
            elif name == "list_files":
                files = self.client.list_files(args["directory"])
                return json.dumps(files)
            else:
                return f"Unknown tool: {name}"
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.error("tool_error", tool=name, error=str(e))
            return f"Error executing {name}: {e}"
