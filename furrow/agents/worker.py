from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


WORKER_TOOL_PROMPT = """You are a worker agent in an autonomous coding system called Furrow.

Your job is to implement the assigned task completely and concisely using tool calls.

Available tools:
- read_file(path): Read file contents
- write_file(path, content): Write content to a file (creates or overwrites)
- edit_file(path, old_string, new_string): Replace specific text in a file
- bash(command): Execute a shell command and return stdout/stderr

Rules:
- Work only on the assigned task. Do not refactor unrelated code.
- Make minimal, targeted changes.
- Always prefer targeted edits over full file rewrites when possible.
- Verify changes look correct before finishing.
- Return a JSON summary at the end: {"status": "completed", "summary": "...", "changes": [...]}

IMPORTANT: You MUST use the tool_calls format to execute file operations.
"""


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_iterations = 10

    async def run(self) -> str:
        prompt = f"""{WORKER_TOOL_PROMPT}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}
Working directory: {self.client.settings.workspace}
"""
        return await self._run_with_tools(prompt)

    async def _run_with_tools(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]
        results: list[dict[str, Any]] = []
        
        for _ in range(self.max_iterations):
            response = await self.client.complete_with_tools(
                messages=messages,
                model=self.client.settings.worker_model,
            )
            
            if not response.content:
                break
                
            text_parts = []
            tool_calls = []
            
            for block in response.content:
                if hasattr(block, 'type'):
                    if block.type == 'text':
                        text_parts.append(block.text)
                    elif block.type == 'tool_use':
                        tool_calls.append({
                            'name': block.name,
                            'input': block.input,
                            'id': block.id,
                        })
            
            if text_parts:
                messages.append({
                    "role": "assistant", 
                    "content": "\n".join(text_parts)
                })
            
            if not tool_calls:
                # No more tool calls, return final response
                return "\n".join(text_parts) if text_parts else "No changes made"
            
            # Execute tool calls
            for tool_call in tool_calls:
                try:
                    result = await self._execute_tool(tool_call["name"], tool_call["input"])
                    results.append({
                        "tool": tool_call["name"],
                        "input": tool_call["input"],
                        "output": result,
                    })
                    messages.append({
                        "role": "user",
                        "content": f"<tool_result id=\"{tool_call['id']}\">{json.dumps(result)}</tool_result>",
                    })
                except Exception as e:
                    error_result = f"Error: {str(e)}"
                    results.append({
                        "tool": tool_call["name"],
                        "input": tool_call["input"],
                        "error": error_result,
                    })
                    messages.append({
                        "role": "user",
                        "content": f"<tool_result id=\"{tool_call['id']}\">{error_result}</tool_result>",
                    })
            
            # Check if any tool call indicates completion
            for tool_call in tool_calls:
                if tool_call["name"] in ("bash", "write_file", "edit_file"):
                    # Check if result contains JSON indicating completion
                    pass
        
        return self._summarize_results(results)

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "read_file":
            return await self._read_file(args.get("path", ""))
        elif name == "write_file":
            return await self._write_file(args.get("path", ""), args.get("content", ""))
        elif name == "edit_file":
            return await self._edit_file(args.get("path", ""), args.get("old_string", ""), args.get("new_string", ""))
        elif name == "bash":
            return await self._bash(args.get("command", ""))
        else:
            return {"error": f"Unknown tool: {name}"}

    async def _read_file(self, path: str) -> dict[str, Any]:
        workspace = self.client.settings.workspace
        full_path = workspace / path if not Path(path).is_absolute() else Path(path)
        try:
            content = await self.client.read_file(full_path)
            return {
                "exists": True,
                "content": content[:5000],  # Limit content size
                "size": len(content),
            }
        except FileNotFoundError:
            return {"exists": False, "error": "File not found"}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    async def _write_file(self, path: str, content: str) -> dict[str, Any]:
        workspace = self.client.settings.workspace
        full_path = workspace / path if not Path(path).is_absolute() else Path(path)
        try:
            await self.client.write_file(full_path, content)
            return {
                "success": True,
                "path": str(full_path),
                "bytes_written": len(content),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _edit_file(self, path: str, old_string: str, new_string: str) -> dict[str, Any]:
        workspace = self.client.settings.workspace
        full_path = workspace / path if not Path(path).is_absolute() else Path(path)
        try:
            content = await self.client.read_file(full_path)
            if old_string not in content:
                return {
                    "success": False, 
                    "error": "old_string not found in file",
                    "hint": "Make sure the old_string matches exactly, including whitespace"
                }
            new_content = content.replace(old_string, new_string, 1)
            await self.client.write_file(full_path, new_content)
            return {
                "success": True,
                "path": str(full_path),
                "bytes_changed": len(new_content) - len(content),
            }
        except FileNotFoundError:
            return {"success": False, "error": "File not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _bash(self, command: str) -> dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *["bash", "-c", command],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.client.settings.workspace),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            return {
                "returncode": proc.returncode,
                "stdout": stdout.decode()[:5000],
                "stderr": stderr.decode()[:2000],
            }
        except asyncio.TimeoutError:
            return {"error": "Command timed out after 120 seconds"}
        except Exception as e:
            return {"error": str(e)}

    def _summarize_results(self, results: list[dict[str, Any]]) -> str:
        changes = []
        errors = []
        for r in results:
            if "error" in r:
                errors.append(f"{r['tool']}: {r['error']}")
            elif r["tool"] == "read_file":
                changes.append(f"Read: {r.get('input', {}).get('path', 'unknown')}")
            elif r["tool"] == "write_file":
                changes.append(f"Wrote: {r.get('path', 'unknown')} ({r.get('bytes_written', 0)} bytes)")
            elif r["tool"] == "edit_file":
                if r.get("success"):
                    changes.append(f"Edited: {r.get('path', 'unknown')} ({r.get('bytes_changed', 0)} bytes)")
                else:
                    errors.append(f"Edit failed: {r.get('error')}")
            elif r["tool"] == "bash":
                rc = r.get("returncode", -1)
                if rc == 0:
                    changes.append(f"Ran: {r.get('stdout', '').strip()[:100]}")
                else:
                    errors.append(f"Bash failed (rc={rc}): {r.get('stderr', '')[:200]}")
        
        summary = []
        if changes:
            summary.append("Changes made:")
            for c in changes[:10]:
                summary.append(f"  • {c}")
        if errors:
            summary.append("Errors:")
            for e in errors[:5]:
                summary.append(f"  • {e}")
        
        return "\n".join(summary) if summary else "No changes made"
