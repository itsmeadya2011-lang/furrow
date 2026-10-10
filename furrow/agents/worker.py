from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import configure_logging, get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

configure_logging()
logger = get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_rounds = 3

    async def run(self) -> str:
        logger.info("worker_started", task_id=self.task.id, description=self.task.description[:50])
        for round_num in range(self.max_rounds):
            prompt = self._build_prompt(round_num)
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                logger.warning("worker_non_json_response", task_id=self.task.id)
                return response
            
            # Execute actions
            result = await self._execute_actions(data.get("actions", []))
            
            if data.get("complete", False):
                logger.info("worker_completed", task_id=self.task.id, summary=data.get('summary'))
                return f"{data.get('summary', 'Task completed')}\n{result}"
            
            # Continue to next round with context
            if round_num < self.max_rounds - 1:
                prompt += f"\n\nPrevious actions result:\n{result}\nContinue working on the task."
        
        logger.warning("worker_max_rounds_reached", task_id=self.task.id)
        return "Max rounds reached. Task may be incomplete."

    def _build_prompt(self, round_num: int) -> str:
        context = f"Round {round_num + 1}/{self.max_rounds}\n"
        context += f"Task: {self.task.description}\n"
        context += f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        if self.task.result:
            context += f"Previous result: {self.task.result}\n"
        return f"{WORKER_PROMPT}\n\n{context}"

    async def _execute_actions(self, actions: list[dict]) -> str:
        logger.debug("worker_executing_actions", task_id=self.task.id, action_count=len(actions))
        results = []
        for action in actions:
            action_type = action.get("type")
            try:
                if action_type == "read":
                    path = action.get("path")
                    content = await self.client.read_file(path)
                    results.append(f"Read {path}: {len(content)} chars")
                elif action_type == "write":
                    path = action.get("path")
                    content = action.get("content", "")
                    await self.client.write_file(path, content)
                    results.append(f"Wrote {path}")
                elif action_type == "list":
                    directory = action.get("directory", ".")
                    files = self.client.list_files(directory)
                    results.append(f"Listed {directory}: {len(files)} files")
                elif action_type == "command":
                    cmd = action.get("cmd", "")
                    result = await self._run_command(cmd)
                    results.append(f"Command '{cmd}': {result}")
                else:
                    results.append(f"Unknown action type: {action_type}")
            except Exception as e:
                logger.error("worker_action_failed", task_id=self.task.id, action_type=action_type, error=str(e))
                results.append(f"Error executing {action_type}: {e}")
        return "\n".join(results)

    async def _run_command(self, cmd: str) -> str:
        logger.debug("worker_running_command", task_id=self.task.id, cmd=cmd)
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.client.settings.workspace
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            output = stdout.decode() + stderr.decode()
            return output[:2000]  # Truncate long output
        except asyncio.TimeoutError:
            logger.warning("worker_command_timeout", task_id=self.task.id, cmd=cmd)
            return "Command timed out"
        except Exception as e:
            logger.error("worker_command_error", task_id=self.task.id, cmd=cmd, error=str(e))
            return f"Command error: {e}"
