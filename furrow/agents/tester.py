from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.max_fix_attempts = 2

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            test_result = TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            test_result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])

        if not test_result.passed and test_result.failures:
            # Try to fix failures
            for attempt in range(self.max_fix_attempts):
                fix_result = await self._fix_failures(goal, test_result.failures)
                if fix_result.passed:
                    return fix_result
                test_result = fix_result
        
        return test_result

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

    async def _fix_failures(self, goal: str, failures: list[str]) -> TestResult:
        """Attempt to fix test failures by analyzing and applying fixes."""
        fix_prompt = f"""You are a tester agent in Furrow. Tests have failed. Analyze the failures and provide fixes.

Goal: {goal}

Test Failures:
{chr(10).join(f"- {f}" for f in failures)}

Available tools:
- read_file(path): Read a file's contents
- write_file(path, content): Write content to a file
- list_files(directory): List files in a directory

Respond with JSON tool calls to apply fixes, or a summary if no fixes can be applied:
[
  {{"tool": "read_file", "args": {{"path": "src/file.py"}}}},
  {{"tool": "write_file", "args": {{"path": "src/file.py", "content": "fixed code"}}}}
]"""
        
        response = await self.client.complete(fix_prompt, model=self.client.settings.tester_model)
        
        # Parse and execute tool calls
        tool_calls = self._parse_tool_calls(response)
        if tool_calls:
            for tool_call in tool_calls:
                await self._execute_tool(tool_call)
            
            # Re-run tests after fixes
            test_output = await self._run_tests()
            
            # Evaluate again
            eval_prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output after fixes:\n{test_output}\n"
            eval_response = await self.client.complete(eval_prompt, model=self.client.settings.tester_model)
            try:
                data = json.loads(eval_response)
                return TestResult(**data)
            except (json.JSONDecodeError, ValueError):
                return TestResult(passed="passed" in eval_response.lower(), summary=eval_response, failures=[])
        
        # No fixes applied, return current result
        return TestResult(passed=False, summary="Could not auto-fix failures", failures=failures)

    def _parse_tool_calls(self, response: str) -> list[dict] | None:
        response = response.strip()
        if response.startswith('['):
            try:
                calls = json.loads(response)
                if isinstance(calls, list) and all(isinstance(c, dict) and 'tool' in c for c in calls):
                    return calls
            except json.JSONDecodeError:
                pass
        return None

    async def _execute_tool(self, tool_call: dict) -> str:
        tool = tool_call.get("tool")
        args = tool_call.get("args", {})
        
        try:
            if tool == "read_file":
                content = await self.client.read_file(args["path"])
                return f"read_file({args['path']}): {content[:500]}{'...' if len(content) > 500 else ''}"
            elif tool == "write_file":
                await self.client.write_file(args["path"], args["content"])
                return f"write_file({args['path']}): OK ({len(args['content'])} chars)"
            elif tool == "list_files":
                files = self.client.list_files(args["path"])
                return f"list_files({args['path']}): {', '.join(files)}"
            else:
                return f"Unknown tool: {tool}"
        except Exception as e:
            return f"Error executing {tool}: {e}"
