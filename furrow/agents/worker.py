from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
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

    async def run(self) -> str:
        system_prompt = WORKER_PROMPT
        prompt = f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
        prompt += "You can use the following tools by responding with JSON:\n"
        prompt += '{"tool": "read_file", "path": "path/to/file"}\n'
        prompt += '{"tool": "write_file", "path": "path/to/file", "content": "file content"}\n'
        prompt += '{"tool": "list_files", "directory": "path/to/dir"}\n'
        prompt += '{"tool": "complete", "summary": "Task completed: ..."}\n\n'
        prompt += "Start by exploring the codebase if needed."

        messages = [{"role": "user", "content": prompt}]
        max_turns = 10
        
        for turn in range(max_turns):
            response = await self.client.complete(
                json.dumps(messages),
                system=system_prompt,
                model=self.client.settings.worker_model
            )
            
            try:
                # Try to parse as JSON tool call
                tool_call = json.loads(response)
                tool_name = tool_call.get("tool")
                
                if tool_name == "read_file":
                    path = tool_call.get("path", "")
                    content = await self.client.read_file(path)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"File content:\n{content}"})
                    logger.info("worker_read_file", task_id=self.task.id, path=path)
                    
                elif tool_name == "write_file":
                    path = tool_call.get("path", "")
                    content = tool_call.get("content", "")
                    await self.client.write_file(path, content)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"File written: {path}"})
                    logger.info("worker_write_file", task_id=self.task.id, path=path)
                    
                elif tool_name == "list_files":
                    directory = tool_call.get("directory", ".")
                    files = self.client.list_files(directory)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Files in {directory}:\n" + "\n".join(files)})
                    logger.info("worker_list_files", task_id=self.task.id, directory=directory)
                    
                elif tool_name == "complete":
                    summary = tool_call.get("summary", "Task completed")
                    logger.info("worker_complete", task_id=self.task.id, summary=summary)
                    return summary
                    
                else:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Unknown tool: {tool_name}. Available: read_file, write_file, list_files, complete"})
                    
            except json.JSONDecodeError:
                # Not a tool call, treat as regular response
                messages.append({"role": "assistant", "content": response})
                # Check if it looks like a completion
                if "complete" in response.lower() or "done" in response.lower():
                    logger.info("worker_complete_implicit", task_id=self.task.id)
                    return response
                messages.append({"role": "user", "content": "Continue working. Use tools by responding with JSON."})
        
        logger.warning("worker_max_turns_reached", task_id=self.task.id)
        return "Max turns reached. Task may be incomplete."
