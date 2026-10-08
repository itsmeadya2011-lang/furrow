from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        file_contents = await self._read_context_files()
        prompt = self._build_prompt(file_contents)
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        result = self._parse_response(response)
        await self._write_files(result["files"])
        return result["summary"]

    async def _read_context_files(self) -> dict[str, str]:
        files: dict[str, str] = {}
        paths_to_read: list[Path] = []

        # Read explicitly listed files
        for path_str in self.task.files:
            path = Path(path_str)
            if path.exists():
                paths_to_read.append(path)

        # If no files specified, discover project files
        if not paths_to_read:
            for f in self.client.list_files("."):
                fp = Path(f)
                if fp.suffix in (".py", ".js", ".ts", ".go", ".rs", ".md", ".json", ".yaml", ".yml", ".toml"):
                    paths_to_read.append(fp)

        # Read files and find related test files
        for path in paths_to_read:
            if str(path) not in files:
                files[str(path)] = await self.client.read_file(path)
            for rel_path in self._find_related_files(path):
                if str(rel_path) not in files and rel_path.exists():
                    files[str(rel_path)] = await self.client.read_file(rel_path)

        return files

    def _find_related_files(self, path: Path) -> list[Path]:
        related: list[Path] = []
        if "test" not in path.parts and "tests" not in path.parts:
            # Look for test files alongside the source file
            stem = path.stem
            candidates = [
                Path("tests") / f"test_{stem}.py",
                Path("tests") / path.name,
                path.parent / f"test_{path.name}",
                path.parent / f"test_{stem}.py",
            ]
            for cand in candidates:
                if cand.exists() and cand not in related:
                    related.append(cand)
        return related

    def _build_prompt(self, file_contents: dict[str, str]) -> str:
        context = ""
        if file_contents:
            context = "\n\n--- Existing Files ---\n"
            for path, content in file_contents.items():
                context += f"\n### {path}\n```\n{content}\n```\n"
        return f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{context}"

    def _parse_response(self, response: str) -> dict:
        try:
            data = json.loads(response.strip())
            if "files" not in data or "summary" not in data:
                raise ValueError("Invalid response format")
            return data
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to parse worker response: {e}\nResponse: {response[:500]}")

    async def _write_files(self, files: list[dict[str, str]]) -> None:
        for file_info in files:
            path = file_info["path"]
            content = file_info["content"]
            await self.client.write_file(path, content)