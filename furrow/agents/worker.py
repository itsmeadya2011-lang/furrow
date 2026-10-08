from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings
        self.workspace = settings.workspace if settings else Path.cwd()
        self.max_rounds = 3

    async def run(self) -> str:
        """Execute the task using LLM with tool access."""
        logger.info("worker_start", task_id=self.task.id, description=self.task.description)
        # Provide context about relevant files
        context = await self._gather_context()

        system_prompt = WORKER_PROMPT
        prompt = f"""{system_prompt}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}

Current file context:
{context}

You have access to the following tools:
- read_file(path): Read a file's contents
- write_file(path, content): Write content to a file
- run_command(cmd, cwd): Run a shell command and return stdout/stderr
- list_files(directory): List all files in a directory

Return a JSON object with your actions:
{{
  "actions": [
    {{"tool": "read_file", "path": "src/file.py"}},
    {{"tool": "write_file", "path": "src/file.py", "content": "new content"}},
    {{"tool": "run_command", "cmd": ["pytest", "-q"], "cwd": "."}}
  ],
  "summary": "Brief summary of what you did"
}}

You can make up to {self.max_rounds} rounds of tool calls. After each round, you'll receive the results and can continue.
"""
        
        # Run the agent loop
        for round_num in range(self.max_rounds):
            logger.debug("worker_round", task_id=self.task.id, round=round_num + 1)
            response = await self.client.complete(
                prompt, 
                model=self.client.settings.worker_model,
                system="You are a coding agent. Return only valid JSON."
            )
            
            try:
                data = json.loads(response)
                actions = data.get("actions", [])
                summary = data.get("summary", "")
                
                if not actions:
                    logger.info("worker_done", task_id=self.task.id, summary=summary)
                    return summary or "No actions taken"
                
                # Execute actions
                results = []
                for action in actions:
                    result = await self._execute_action(action)
                    results.append(f"{action.get('tool')}: {result}")
                    logger.debug("worker_action", task_id=self.task.id, tool=action.get('tool'), result=result[:100])
                
                # If this is the last round or no more actions needed, return summary
                if round_num == self.max_rounds - 1:
                    final_result = summary + "\n" + "\n".join(results)
                    logger.info("worker_complete", task_id=self.task.id, rounds=round_num + 1)
                    return final_result
                
                # Continue with results as context
                prompt = f"""Previous actions results:
{chr(10).join(results)}

Continue with next actions if needed, or return empty actions with summary if done.
Return JSON only."""
                
            except json.JSONDecodeError:
                # If not valid JSON, treat as final response
                logger.info("worker_response_text", task_id=self.task.id, response=response[:100])
                return response

        logger.warning("worker_max_rounds", task_id=self.task.id)
        return "Max rounds reached"

    async def _gather_context(self) -> str:
        """Gather context about relevant files."""
        context_parts = []
        
        # Read specified files
        for file_path in self.task.files:
            full_path = self.workspace / file_path
            if full_path.exists():
                try:
                    content = await self.client.read_file(full_path)
                    context_parts.append(f"=== {file_path} ===\n{content}")
                except (OSError, UnicodeDecodeError) as e:
                    logger.debug("read_file_error", file=file_path, error=str(e))
                    context_parts.append(f"=== {file_path} === (could not read)")
        
        # Also list workspace files for reference
        files = self.client.list_files(self.workspace)
        context_parts.append(f"Workspace files: {', '.join(files[:50])}")
        
        return "\n\n".join(context_parts) if context_parts else "No context available"

    async def _execute_action(self, action: dict[str, Any]) -> str:
        """Execute a single tool action."""
        tool = action.get("tool")
        
        try:
            if tool == "read_file":
                path = action.get("path")
                if not path:
                    return "Error: path required"
                full_path = self.workspace / path
                content = await self.client.read_file(full_path)
                return content[:2000]  # Truncate long files
                
            elif tool == "write_file":
                path = action.get("path")
                content = action.get("content", "")
                if not path:
                    return "Error: path required"
                full_path = self.workspace / path
                await self.client.write_file(full_path, content)
                return f"Written {len(content)} bytes to {path}"
                
            elif tool == "run_command":
                cmd = action.get("cmd")
                cwd = action.get("cwd", ".")
                if not cmd:
                    return "Error: cmd required"
                if isinstance(cmd, str):
                    cmd = cmd.split()
                full_cwd = self.workspace / cwd
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=full_cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                output = stdout.decode() + stderr.decode()
                return output[:2000] if output else f"Command exited with code {proc.returncode}"
                
            elif tool == "list_files":
                directory = action.get("directory", ".")
                full_dir = self.workspace / directory
                files = self.client.list_files(full_dir)
                return ", ".join(files)
                
            else:
                return f"Unknown tool: {tool}"
                
        except asyncio.TimeoutError:
            return "Error: Command timed out"
        except (OSError, ValueError, RuntimeError) as e:
            logger.debug("action_error", tool=tool, error=str(e))
            return f"Error: {e}"
