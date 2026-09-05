"""Base agent class with an LLM tool-use loop."""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Sequence

from furrow.llm import LLMClient, LLMResult, ToolCall
from furrow.logging import get_logger
from furrow.tools.base import Tool

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger(__name__)


class Agent:
    """Base class for LLM agents with tool-use capability.

    Subclasses set ``system_prompt`` and ``tools``, then call
    :meth:`run_loop` to execute an iterative LLM conversation.
    """

    system_prompt: str = ""
    tools: list[type[Tool]] = []

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        if client is not None:
            self.client = client
        else:
            self.client = LLMClient(settings=settings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        prompt: str,
        model: str | None = None,
        max_iterations: int = 10,
    ) -> str:
        """Run the agent loop and return the final text response."""
        result = await self.run_loop(prompt, model=model, max_iterations=max_iterations)
        return result.text

    async def run_loop(
        self,
        prompt: str,
        model: str | None = None,
        max_iterations: int = 10,
    ) -> LLMResult:
        """Run an LLM conversation with tool-use.

        The loop proceeds as:
        1. Send prompt + tools to the LLM.
        2. If the LLM requests tool calls, execute them.
        3. Feed results back and repeat, up to ``max_iterations``.
        """
        model = model or self.client.settings.worker_model
        tool_defs = [t.to_anthropic_schema() for t in self.tools] if self.tools else None

        messages: list[dict[str, Any]] = []
        result: LLMResult | None = None

        for iteration in range(1, max_iterations + 1):
            log.debug(
                "Agent loop iteration",
                iteration=iteration,
                has_prompt=bool(prompt),
                messages=len(messages),
            )

            result = await self.client.complete_with_tools(
                prompt=prompt,
                system=self.system_prompt,
                model=model,
                tools=tool_defs,
                messages=messages if messages else None,
            )

            if not result.tool_calls:
                # No more tools requested — we're done
                log.debug("Agent loop ended (no tool calls)")
                return result

            # Add the assistant's response to the conversation
            content_blocks: list[dict[str, Any]] = []
            if result.text:
                content_blocks.append({"type": "text", "text": result.text})
            content_blocks.extend(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                }
                for tc in result.tool_calls
            )
            messages.append({"role": "assistant", "content": content_blocks})

            # Execute all tool calls concurrently
            tool_results = await self._execute_tools(result.tool_calls)
            log.debug("Tools executed", results=len(tool_results))

            # Format tool results as a user message
            result_blocks: list[dict[str, Any]] = []
            for tc, tool_out in zip(result.tool_calls, tool_results):
                result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": tool_out,
                })
            messages.append({"role": "user", "content": result_blocks})
            prompt = ""  # subsequent iterations use messages only

        log.warning("Agent loop hit max iterations without finishing", max_iterations=max_iterations)
        return result or LLMResult(text="Max iterations reached")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _execute_tools(self, tool_calls: Sequence[ToolCall]) -> list[str]:
        """Execute a sequence of tool calls concurrently and return outputs."""
        tool_map = {tool_cls.name: tool_cls for tool_cls in self.tools}
        from furrow.tools.base import ToolResult

        async def _call(tc: ToolCall) -> str:
            tool_cls = tool_map.get(tc.name)
            if tool_cls is None:
                return json.dumps({"error": f"Unknown tool: {tc.name}"})
            try:
                out = await tool_cls.execute(**tc.input)
                if isinstance(out, str):
                    return out
                if isinstance(out, ToolResult):
                    if out.success:
                        return out.output or ""
                    return f"Error: {out.error}" if out.error else out.output
                return str(out)
            except TypeError as e:
                return json.dumps({"error": f"Tool argument mismatch: {e}"})
            except Exception as e:
                log.error("Tool execution failed", tool=tc.name, error=str(e))
                return json.dumps({"error": str(e)})

        results = await asyncio.gather(*[_call(tc) for tc in tool_calls])
        return list(results)
