from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        max_fix_rounds: int = 3,
    ) -> None:
        self.client = client or LLMClient(settings=settings)
        self.max_fix_rounds = max_fix_rounds
    
    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run tests and attempt to fix failures."""
        # First, run tests to see current state
        test_output = await self._run_tests()
        
        # Check if tests already pass
        if self._tests_passed(test_output):
            return TestResult(passed=True, summary="All tests passed", failures=[])
        
        # Try to fix failures
        for round_num in range(self.max_fix_rounds):
            console_print(f"[yellow]Test fix attempt {round_num + 1}/{self.max_fix_rounds}[/yellow]")
            
            # Get fix from LLM with tool use
            fix_result = await self._attempt_fix(goal, tasks, test_output)
            
            if not fix_result:
                break
            
            # Re-run tests
            test_output = await self._run_tests()
            
            if self._tests_passed(test_output):
                return TestResult(passed=True, summary="Tests fixed and passing", failures=[])
        
        # Final test run
        final_output = await self._run_tests()
        passed = self._tests_passed(final_output)
        
        if passed:
            return TestResult(passed=True, summary="Tests passing after fixes", failures=[])
        
        failures = self._extract_failures(final_output)
        return TestResult(passed=False, summary="Tests still failing after fix attempts", failures=failures)
    
    async def _attempt_fix(self, goal: str, tasks: list[TaskModel], test_output: str) -> bool:
        """Attempt to fix failing tests using tool-use loop."""
        system_prompt = TESTER_PROMPT
        
        # Build context with relevant files from tasks
        context = await self._build_context(tasks)
        
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n\nContext:\n{context}\n\nFix the failing tests. Use tools to read and modify files."}
        ]
        
        for round_num in range(self.max_fix_rounds):
            response = await self.client.complete(
                "\n".join(m["content"] for m in messages),
                system=system_prompt,
                model=self.client.settings.tester_model,
            )
            
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # No tool calls, assume fix is complete or LLM gave up
                return False
            
            # Execute tool calls
            for tool_call in tool_calls:
                result = await self._execute_tool(tool_call)
                messages.append({"role": "tool", "content": json.dumps({"result": result})})
            
            messages.append({"role": "assistant", "content": response})
        
        return True
    
    async def _build_context(self, tasks: list[TaskModel]) -> str:
        """Build context from task files."""
        context_parts = []
        seen_files = set()
        
        for task in tasks:
            for file_path in task.files:
                if file_path in seen_files:
                    continue
                seen_files.add(file_path)
                try:
                    content = await self.client.read_file(file_path)
                    context_parts.append(f"--- {file_path} ---\n{content}")
                except FileNotFoundError:
                    context_parts.append(f"--- {file_path} ---\n[File does not exist]")
                except Exception as e:
                    context_parts.append(f"--- {file_path} ---\n[Error reading: {e}]")
        
        return "\n\n".join(context_parts) if context_parts else "No files found."
    
    def _tests_passed(self, output: str) -> bool:
        """Check if test output indicates all tests passed."""
        output_lower = output.lower()
        # Common patterns for passing tests
        if "passed" in output_lower and "failed" not in output_lower and "error" not in output_lower:
            return True
        if "ok" in output_lower and "failed" not in output_lower:
            return True
        if "all tests passed" in output_lower:
            return True
        return False
    
    def _extract_failures(self, output: str) -> list[str]:
        """Extract failure messages from test output."""
        failures = []
        lines = output.split("\n")
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ["failed", "error", "fail:", "assertion", "traceback"]):
                if line.strip():
                    failures.append(line.strip())
        
        return failures[:10]  # Limit to 10 failures
    
    def _parse_tool_calls(self, response: str) -> list[dict[str, Any]]:
        """Parse tool calls from LLM response."""
        tool_calls = []
        pattern = r"```tool\n(\{.*?\})\n```"
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        
        return tool_calls
    
    async def _execute_tool(self, tool_call: dict[str, Any]) -> str:
        """Execute a tool call and return the result."""
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})
        
        try:
            if name == "read_file":
                path = args.get("path", "")
                if not path:
                    return "Error: path is required"
                return await self.client.read_file(path)
            
            elif name == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                if not path:
                    return "Error: path is required"
                await self.client.write_file(path, content)
                return f"Successfully wrote to {path}"
            
            elif name == "list_files":
                directory = args.get("directory", ".")
                files = self.client.list_files(directory)
                return json.dumps(files, indent=2)
            
            elif name == "run_command":
                cmd = args.get("command", "")
                if not cmd:
                    return "Error: command is required"
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                return f"Exit code: {proc.returncode}\nStdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}"
            
            else:
                return f"Error: Unknown tool '{name}'"
        
        except Exception as e:
            return f"Error executing {name}: {e}"
    
    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."


# Helper to avoid circular import
def console_print(msg: str) -> None:
    from rich.console import Console
    Console().print(msg)