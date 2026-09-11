from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("furrow.worker")


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_rounds = 3

    async def run(self) -> str:
        system = WORKER_PROMPT
        prompt = f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        
        tools = self.client.get_tool_definitions()
        rounds = 0
        
        while rounds < self.max_rounds:
            rounds += 1
            logger.debug("worker_round", task_id=self.task.id, round=rounds)
            response_text, tool_calls = await self.client.complete_with_tools(
                prompt, system, model=self.client.settings.worker_model, tools=tools
            )
            
            if not tool_calls:
                # No tool calls, task is complete
                logger.info("worker_done", task_id=self.task.id, rounds=rounds)
                return response_text or "Task completed (no tool calls made)"
            
            # Execute tool calls
            results = []
            for tool_call in tool_calls:
                try:
                    logger.debug("tool_call", task_id=self.task.id, tool=tool_call["name"])
                    result = await self.client.execute_tool(tool_call["name"], tool_call["arguments"])
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "result": str(result),
                    })
                except Exception as e:
                    logger.error("tool_error", task_id=self.task.id, tool=tool_call["name"], error=str(e))
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "result": f"Error: {e}",
                    })
            
            # Feed results back to the model
            prompt = self._format_tool_results(results)
        
        logger.warning("worker_max_rounds", task_id=self.task.id, rounds=rounds)
        return f"Task completed after {rounds} rounds. Final response: {response_text}"

    def _format_tool_results(self, results: list[dict]) -> str:
        """Format tool results for the next LLM call."""
        parts = ["Tool results:"]
        for r in results:
            parts.append(f"  Tool call {r['tool_call_id']}: {r['result']}")
        parts.append("\nContinue working on the task. If done, provide a summary without tool calls.")
        return "\n".join(parts)
