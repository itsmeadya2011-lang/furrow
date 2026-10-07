from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # First, check if tests pass using the LLM
        prompt = f"""{TESTER_PROMPT}

Goal: {goal}

Test output:
{test_output}
"""
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)

        try:
            data = json.loads(response)
            initial_result = TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Fallback: check if "passed" appears in response
            initial_result = TestResult(
                passed="passed" in response.lower() and "fail" not in response.lower(),
                summary=response,
                failures=[]
            )

        # If tests failed, attempt to fix them
        if not initial_result.passed and initial_result.failures:
            fixed = await self._attempt_fix(goal, test_output, initial_result.failures)
            if fixed:
                # Re-run tests after fixing
                try:
                    new_output = await self._run_tests()
                    # Check if fixed tests now pass
                    recheck_prompt = f"""{TESTER_PROMPT}

Goal: {goal}

Test output after fix:
{new_output}
"""
                    recheck_response = await self.client.complete(recheck_prompt, model=self.client.settings.tester_model)
                    try:
                        recheck_data = json.loads(recheck_response)
                        return TestResult(**recheck_data)
                    except (json.JSONDecodeError, ValueError):
                        return TestResult(
                            passed="passed" in recheck_response.lower() and "fail" not in recheck_response.lower(),
                            summary=recheck_response,
                            failures=[]
                        )
                except Exception as e:
                    return TestResult(passed=False, summary=f"Fix applied but re-test failed: {e}", failures=[str(e)])

        return initial_result

    async def _attempt_fix(self, goal: str, test_output: str, failures: list[str]) -> bool:
        """Ask the LLM to diagnose and fix test failures. Returns True if fixes were applied."""
        # Collect all files mentioned in the failures or touched by tasks
        files_to_check = set()
        for failure in failures:
            # Extract file paths from failure messages (common patterns)
            for match in re.finditer(r'[\w/]+\.(py|js|ts|go|rs)', failure):
                files_to_check.add(match.group(0))

        context_parts = []
        for filepath in files_to_check:
            try:
                content = await self.client.read_file(filepath)
                context_parts.append(f"--- {filepath} ---\n{content}\n")
            except FileNotFoundError:
                pass

        context = "\n".join(context_parts) if context_parts else "(no file context available)"

        fix_prompt = f"""You are a debugging expert. Tests are failing and you need to fix the code.

Goal: {goal}

Failures:
{chr(10).join(f'- {f}' for f in failures)}

Test output:
{test_output}

Relevant files:
{context}

Return a JSON object with the fixes to apply:
{{
  "files": {{
    "path/to/file.py": "full corrected file content"
  }},
  "explanation": "what was wrong and how you fixed it"
}}

Only include files that need changes. Return valid JSON only, no markdown.
"""
        response = await self.client.complete(fix_prompt, model=self.client.settings.tester_model)

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    return False
            else:
                return False

        files = data.get("files", {})
        if not files:
            return False

        for filepath, content in files.items():
            await self.client.write_file(filepath, content)

        return True

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