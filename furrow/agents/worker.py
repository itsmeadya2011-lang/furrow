from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Provider, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 25

# Anthropic tool definitions
ANTHROPIC_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read the contents of a file and return its text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to read.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write or create a file. Creates parent directories as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The full content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Execute a shell command and return its combined stdout and stderr output. "
            "Use this to run tests, linters, or typecheckers to verify changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "The shell command to execute.",
                }
            },
            "required": ["cmd"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files under a directory (relative paths).",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "The directory to list files under.",
                }
            },
            "required": ["directory"],
        },
    },
]

# OpenAI-compatible tool definitions
OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in ANTHROPIC_TOOLS
]


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        provider = self.client.settings.provider
        if provider == Provider.ANTHROPIC:
            return await self._run_anthropic(prompt)
        elif provider in (Provider.OPENAI, Provider.OLLAMA):
            return await self._run_openai(prompt)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == "read_file":
            return await self.client.read_file(args["path"])
        elif name == "write_file":
            await self.client.write_file(args["path"], args["content"])
            return f"Wrote {args['path']}"
        elif name == "run_command":
            return await self.client.run_command(args["cmd"])
        elif name == "list_files":
            files = self.client.list_files(args["directory"])
            return "\n".join(files) if files else "(no files)"
        else:
            return f"Unknown tool: {name}"

    async def _run_anthropic(self, prompt: str) -> str:
        model = self.client.settings.worker_model
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        final_text = ""

        for _ in range(MAX_TOOL_ROUNDS):
            response = await self.client.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=WORKER_PROMPT,
                messages=messages,
                tools=ANTHROPIC_TOOLS,
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if text_blocks:
                final_text = text_blocks[-1].text

            if not tool_uses:
                return final_text or ""

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tu in tool_uses:
                result = await self._execute_tool(tu.name, tu.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        log.warning("anthropic_tool_rounds_exhausted", max=MAX_TOOL_ROUNDS)
        return final_text or ""

    async def _run_openai(self, prompt: str) -> str:
        client = self.client.openai if self.client.settings.provider == Provider.OPENAI else self.client.ollama
        model = self.client.settings.worker_model
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": WORKER_PROMPT},
            {"role": "user", "content": prompt},
        ]
        final_text = ""

        for _ in range(MAX_TOOL_ROUNDS):
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=OPENAI_TOOLS,
            )

            choice = response.choices[0]
            msg = choice.message

            if msg.content:
                final_text = msg.content

            if not msg.tool_calls:
                return final_text or ""

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = await self._execute_tool(tc.function.name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        log.warning("openai_tool_rounds_exhausted", max=MAX_TOOL_ROUNDS)
        return final_text or ""