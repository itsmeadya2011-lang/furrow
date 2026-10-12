from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.agents.tools import create_tool, get_tool_descriptions, TOOLS
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        max_rounds: int = 10
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_rounds = max_rounds
        self.tools = {name: create_tool(name, self.client) for name in TOOLS}

    async def run(self) -> str:
        system_prompt = f"""{WORKER_PROMPT}

You have access to the following tools:
{get_tool_descriptions()}

To use a tool, respond with a JSON object in this format:
```json
{{
  "tool": "tool_name",
  "arguments": {{"arg1": "value1", "arg2": "value2"}}
}}
```

You can make multiple tool calls in sequence. After each tool call, you'll receive the result and can decide what to do next.

When you're done, respond with a final summary (not JSON) describing what you accomplished.
"""

        initial_prompt = f"""Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}
Dependencies: {', '.join(self.task.dependencies) if self.task.dependencies else 'none'}

Start working on this task now."""

        messages: list[dict] = [
            {"role": "user", "content": initial_prompt}
        ]

        for round_num in range(self.max_rounds):
            response = await self.client.complete(
                system=system_prompt,
                model=self.client.settings.worker_model,
                messages=messages
            )

            # Try to parse tool calls from response
            tool_calls = self._extract_tool_calls(response)
            if not tool_calls:
                # No tool calls, treat as final response
                return response

            # Execute tool calls
            messages.append({"role": "assistant", "content": response})
            for tool_call in tool_calls:
                tool_name = tool_call.get("tool")
                arguments = tool_call.get("arguments", {})
                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    result = await tool.execute(**arguments)
                    messages.append({"role": "user", "content": f"Tool result:\n{result}"})
                else:
                    messages.append({"role": "user", "content": f"Error: Unknown tool '{tool_name}'"})

        return f"Max rounds ({self.max_rounds}) reached. Last response: {response}"

    def _extract_tool_calls(self, response: str) -> list[dict]:
        """Extract tool calls from the response."""
        # Look for JSON code blocks
        pattern = r'```json\s*(\{.*?\})\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        calls = []
        for match in matches:
            try:
                data = json.loads(match)
                if "tool" in data:
                    calls.append(data)
            except json.JSONDecodeError:
                continue
        return calls