from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT, WORKER_SYSTEM_PROMPT
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
        logger.info("worker_started", task_id=self.task.id, description=self.task.description[:100])
        # Read relevant files
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = ""  # New file

        # Build prompt with file contents
        files_section = "\n\n".join(
            f"=== {path} ===\n{content}" if content else f"=== {path} === (new file)"
            for path, content in file_contents.items()
        )

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n\nFiles:\n{files_section}"
        
        # Get diffs from LLM
        diffs_output = await self.client.complete(prompt, system=WORKER_SYSTEM_PROMPT, model=self.client.settings.worker_model)
        
        # Parse and apply diffs
        applied_files = await self._apply_diffs(diffs_output)
        
        result = f"Modified {len(applied_files)} file(s): {', '.join(applied_files)}"
        logger.info("worker_completed", task_id=self.task.id, modified_files=applied_files)
        return result

    async def _apply_diffs(self, diffs_output: str) -> list[str]:
        """Parse unified diffs and apply them to files."""
        applied = []
        
        # Split by diff headers
        diff_pattern = r"(--- a/(.+?)\n\+\+\+ b/(.+?)\n@@.*?)(?=\n--- a/|\Z)"
        matches = re.findall(diff_pattern, diffs_output, re.DOTALL)
        
        for _, old_path, new_path in matches:
            file_path = new_path.strip()
            if file_path == "/dev/null":
                continue  # Deletion not supported yet
            
            # Extract the diff content for this file
            diff_start = diffs_output.find(f"--- a/{old_path}")
            if diff_start == -1:
                continue
            diff_end = diffs_output.find(f"\n--- a/", diff_start + 1)
            if diff_end == -1:
                diff_end = len(diffs_output)
            file_diff = diffs_output[diff_start:diff_end].strip()
            
            # Read current content
            try:
                current_content = await self.client.read_file(file_path)
            except FileNotFoundError:
                current_content = ""
            
            # Apply diff
            new_content = self._apply_unified_diff(current_content, file_diff)
            if new_content != current_content:
                await self.client.write_file(file_path, new_content)
                applied.append(file_path)
        
        return applied

    def _apply_unified_diff(self, content: str, diff: str) -> str:
        """Apply a unified diff to content."""
        lines = content.splitlines(keepends=True)
        diff_lines = diff.splitlines(keepends=True)
        
        # Parse the diff header to find line numbers
        # Format: @@ -old_start,old_count +new_start,new_count @@
        import re
        hunk_header = re.search(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@", diff)
        if not hunk_header:
            return content
        
        old_start = int(hunk_header.group(1)) - 1  # Convert to 0-indexed
        old_count = int(hunk_header.group(2)) if hunk_header.group(2) else 1
        new_start = int(hunk_header.group(3)) - 1
        # new_count = int(hunk_header.group(4)) if hunk_header.group(4) else 1
        
        # Parse hunks
        i = 0
        while i < len(diff_lines):
            line = diff_lines[i]
            if line.startswith("@@"):
                # Parse new hunk
                hunk_match = re.search(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@", line)
                if hunk_match:
                    old_start = int(hunk_match.group(1)) - 1
                    old_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
                i += 1
                continue
            
            if line.startswith(" "):
                # Context line - skip
                i += 1
            elif line.startswith("-"):
                # Removal
                if 0 <= old_start < len(lines):
                    lines.pop(old_start)
                i += 1
            elif line.startswith("+"):
                # Addition
                if 0 <= old_start <= len(lines):
                    lines.insert(old_start, line[1:] + "\n" if not line[1:].endswith("\n") else line[1:])
                    old_start += 1
                i += 1
            else:
                i += 1
        
        return "".join(lines)
