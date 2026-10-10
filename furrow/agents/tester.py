from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("tester")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None, workspace: Path | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.workspace = workspace or Path.cwd()
        self.max_fix_attempts = 3

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.info("tester_start", goal=goal, task_count=len(tasks))
        
        for attempt in range(self.max_fix_attempts):
            test_output = ""
            try:
                test_output = await self._run_tests()
            except (OSError, ValueError, RuntimeError) as e:
                logger.error("test_run_error", error=str(e))
                return TestResult(passed=False, summary=str(e), failures=[str(e)])

            # Check if tests passed by looking at output
            if self._tests_passed(test_output):
                logger.info("tests_passed", attempt=attempt)
                return TestResult(passed=True, summary="All tests passed", failures=[])

            # Tests failed - try to fix them
            if attempt < self.max_fix_attempts - 1:
                logger.warning("tests_failed_attempt_fix", attempt=attempt, output_length=len(test_output))
                fix_result = await self._fix_failures(goal, test_output, tasks)
                if not fix_result:
                    break  # Could not fix

        # Final test run
        test_output = await self._run_tests()
        if self._tests_passed(test_output):
            logger.info("tests_fixed_and_passing")
            return TestResult(passed=True, summary="Tests fixed and passing", failures=[])

        # Still failing
        failures = self._extract_failures(test_output)
        logger.error("tests_still_failing", failures=failures)
        return TestResult(passed=False, summary=test_output[:500], failures=failures)

    def _tests_passed(self, output: str) -> bool:
        output_lower = output.lower()
        # Common patterns for test success
        if "passed" in output_lower and "failed" not in output_lower:
            return True
        if "all tests passed" in output_lower:
            return True
        if "ok" in output_lower and "error" not in output_lower and "fail" not in output_lower:
            return True
        # pytest specific
        return bool("passed" in output_lower and "==" in output_lower and "failed" not in output_lower)

    def _extract_failures(self, output: str) -> list[str]:
        failures = []
        lines = output.split("\n")
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in ["error", "fail", "assert", "exception", "traceback"]):
                failures.append(line.strip())
        return failures[:10]  # Limit to 10 failures

    async def _fix_failures(self, goal: str, test_output: str, tasks: list[TaskModel]) -> bool:
        """Try to fix failing tests by analyzing output and making changes."""
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output (FAILED):\n{test_output}\n\nAnalyze the failures and fix the code. Use tools to read/edit files as needed."
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Fix the failing tests. Start by examining the relevant files."},
        ]

        for round_num in range(3):  # Max 3 tool rounds for fixing
            response = await self.client.complete_with_tools(messages, model=self.client.settings.tester_model)
            
            if response.get("tool"):
                tool_result = await self._execute_tool(response["tool"], response.get("args", {}))
                messages.append({"role": "assistant", "content": json.dumps(response)})
                messages.append({"role": "user", "content": f"Tool result:\n{tool_result}"})
            else:
                break  # Done fixing

        return True

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
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                result = f"Exit code: {proc.returncode}\nStdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}"
                logger.info("tool_run_command", cmd=cmd, args=cmd_args, exit_code=proc.returncode)
                return result
            else:
                logger.warning("tool_unknown", tool=tool)
                return f"Unknown tool: {tool}"
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("tool_error", tool=tool, error=str(e))
            return f"Tool error: {e}"

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
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=self.workspace
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    logger.debug("test_run", cmd=cmd[0], exit_code=proc.returncode, output_length=len(output))
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_timeout", cmd=cmd[0])
                    continue
            except (FileNotFoundError, OSError) as e:
                logger.debug("test_runner_not_found", cmd=cmd[0], error=str(e))
                continue
        logger.warning("no_test_runner_found")
        return "No test runner found."
