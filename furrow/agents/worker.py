from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("worker")


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None, workspace: Path | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = workspace or Path.cwd()
        self.max_tool_rounds = 3

    async def run(self) -> str:
        system_prompt = WORKER_PROMPT + """

You have access to the following tools. Use them by responding with JSON in this format:
{
  "tool": "tool_name",
  "args": {...}
}

Available tools:
- read_file(path): Read a file's contents
- write_file(path, content): Write content to a file (creates directories)
- edit_file(path, old_string, new_string): Replace exact string in file
- glob_files(pattern): Find files matching a glob pattern
- grep_files(pattern, include=None): Search file contents with regex
- list_files(directory): List all files in a directory
- run_command(cmd, args...): Run a shell command and return output

After each tool call, you'll receive the result. Continue until the task is complete.
Then respond with a final summary (plain text, not JSON) of what you accomplished."""

        prompt = f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        logger.info("worker_start", task_id=self.task.id, description=self.task.description)

        for round_num in range(self.max_tool_rounds):
            response = await self.client.complete_with_tools(messages, model=self.client.settings.worker_model)
            
            # Check if it's a tool call
            if response.get("tool"):
                tool_name = response["tool"]
                args = response.get("args", {})
                logger.debug("tool_call", task_id=self.task.id, tool=tool_name, args=args)
                tool_result = await self._execute_tool(tool_name, args)
                messages.append({"role": "assistant", "content": json.dumps(response)})
                messages.append({"role": "user", "content": f"Tool result:\n{tool_result}"})
            else:
                # Final response
                content = response.get("content", str(response))
                logger.info("worker_complete", task_id=self.task.id, result_length=len(content))
                return content

        logger.warning("worker_max_rounds", task_id=self.task.id)
        return "Max tool rounds reached. Task may be incomplete."

    async def _execute_tool(self, tool: str, args: dict) -> str:
        try:
            if tool == "read_file":
                path = self.workspace / args["path"]
                content = await self.client.read_file(path)
                logger.debug("tool_read_file", path=args["path"], length=len(content))
                return content
            elif tool == "write_file":
                path = self.workspace / args["path"]
                await self.client.write_file(path, args["content"])
                logger.info("tool_write_file", path=args["path"])
                return f"Written to {args['path']}"
            elif tool == "edit_file":
                path = self.workspace / args["path"]
                content = await self.client.read_file(path)
                old = args["old_string"]
                new = args["new_string"]
                if old not in content:
                    logger.warning("tool_edit_not_found", path=args["path"])
                    return f"Error: old_string not found in {args['path']}"
                new_content = content.replace(old, new, 1)
                await self.client.write_file(path, new_content)
                logger.info("tool_edit_file", path=args["path"])
                return f"Edited {args['path']}"
            elif tool == "glob_files":
                pattern = args["pattern"]
                files = list(self.workspace.glob(pattern))
                result = "\n".join(str(f.relative_to(self.workspace)) for f in files)
                logger.debug("tool_glob", pattern=pattern, count=len(files))
                return result
            elif tool == "grep_files":
                pattern = args["pattern"]
                include = args.get("include")
                results = []
                for file_path in self.workspace.rglob("*"):
                    if file_path.is_file() and (not include or file_path.match(include)):
                        try:
                            content = await self.client.read_file(file_path)
                            if re.search(pattern, content):
                                results.append(str(file_path.relative_to(self.workspace)))
                        except (OSError, UnicodeDecodeError):
                            logger.debug("grep_file_read_failed", path=str(file_path))
                            continue
                result = "\n".join(results)
                logger.debug("tool_grep", pattern=pattern, matches=len(results))
                return result
            elif tool == "list_files":
                directory = self.workspace / args.get("directory", ".")
                files = [str(f.relative_to(self.workspace)) for f in directory.rglob("*") if f.is_file()]
                result = "\n".join(files)
                logger.debug("tool_list_files", directory=args.get("directory", "."), count=len(files))
                return result
            elif tool == "run_command":
                import asyncio
                cmd = args["cmd"]
                cmd_args = args.get("args", [])
                proc = await asyncio.create_subprocess_exec(
                    cmd, *cmd_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.workspace
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                result = f"Exit code: {proc.returncode}\nStdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}"
                logger.info("tool_run_command", cmd=cmd, args=cmd_args, exit_code=proc.returncode)
                return result
            else:
                logger.warning("tool_unknown", tool=tool)
                return f"Unknown tool: {tool}"
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("tool_error", tool=tool, error=str(e))
            return f"Tool error: {e}"
