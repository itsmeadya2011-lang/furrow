from __future__ import annotations

import re
from pathlib import Path

from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings
from furrow.config import settings as global_settings
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)
        self.workspace = Path(workspace) if workspace is not None else Path(self.settings.workspace)

    async def plan(self, goal: str, context: str = "") -> Plan:
        # Search for relevant code files to provide context
        code_context = await self._gather_code_context(goal)
        
        prompt = PLANNER_PROMPT + "\n\nGoal: " + goal
        if code_context:
            prompt += "\n\nRelevant codebase context:\n" + code_context
        if context:
            prompt += "\n\nAdditional context:\n" + context

        model = self.settings.planner_model
        response = await self.client.complete(prompt, system=PLANNER_SYSTEM, model=model)
        try:
            return parse_plan(response)
        except ValueError as first:
            stricter = (
                prompt
                + "\n\nYour previous reply could not be parsed. Reply with ONLY a JSON object "
                + "matching the required shape, no prose, no markdown fences."
            )
            response2 = await self.client.complete(stricter, system=PLANNER_SYSTEM, model=model)
            try:
                return parse_plan(response2)
            except ValueError as second:
                raw = response2 if isinstance(response2, str) else str(response2)
                raise ValueError(
                    "Planner output could not be parsed. "
                    f"attempt 1: {first}. attempt 2: {second}. Response: {raw[:500]}"
                ) from second

    async def _gather_code_context(self, goal: str) -> str:
        """Search the codebase for files relevant to the goal."""
        # Extract keywords from the goal for searching
        keywords = self._extract_keywords(goal)
        if not keywords:
            return ""
        
        # Search for each keyword
        all_matches = []
        seen_files = set()
        
        for keyword in keywords[:5]:  # Limit to top 5 keywords
            matches = await self._search_keyword(keyword)
            for match in matches:
                file_path = match.split(":")[0]
                if file_path not in seen_files:
                    seen_files.add(file_path)
                    all_matches.append(match)
                if len(all_matches) >= 20:  # Limit total matches
                    break
            if len(all_matches) >= 20:
                break
        
        if not all_matches:
            return ""
        
        # Read relevant files and include their content
        file_contents = []
        for match in all_matches[:10]:  # Limit to top 10 files
            file_path = match.split(":")[0]
            try:
                full_path = self.workspace / file_path
                if full_path.exists() and full_path.is_file():
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    # Truncate large files
                    if len(content) > 5000:
                        content = content[:5000] + "\n[... truncated ...]"
                    file_contents.append(f"=== {file_path} ===\n{content}")
            except Exception:
                continue
        
        return "\n\n".join(file_contents)

    def _extract_keywords(self, goal: str) -> list[str]:
        """Extract relevant keywords from the goal for code search."""
        # Remove common words and extract meaningful terms
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can", "need",
            "want", "create", "implement", "add", "make", "build", "write",
            "update", "fix", "change", "modify", "refactor", "improve",
            "goal", "task", "project", "code", "file", "function", "class",
            "module", "api", "test", "tests", "testing"
        }
        
        # Split on non-alphanumeric and filter
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", goal.lower())
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]
        
        # Also extract camelCase and snake_case identifiers
        identifiers = re.findall(r"[a-z]+(?:[A-Z][a-z]+)+|[a-z]+(?:_[a-z]+)+", goal)
        keywords.extend(identifiers)
        
        # Deduplicate and return top keywords
        seen = set()
        unique_keywords = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique_keywords.append(k)
        
        return unique_keywords[:10]

    async def _search_keyword(self, keyword: str) -> list[str]:
        """Search for a keyword in the codebase."""
        matches = []
        try:
            for file_path in self.workspace.rglob("*"):
                if not file_path.is_file():
                    continue
                # Skip ignored directories
                if any(part.startswith(".") for part in file_path.parts):
                    continue
                if any(part in {"node_modules", "__pycache__", "build", "dist", "venv", ".venv", ".git", ".furrow"} for part in file_path.parts):
                    continue
                # Skip large files
                try:
                    if file_path.stat().st_size > 500_000:
                        continue
                except OSError:
                    continue
                
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    for i, line in enumerate(content.splitlines(), 1):
                        if keyword.lower() in line.lower():
                            rel_path = file_path.relative_to(self.workspace)
                            matches.append(f"{rel_path}:{i}: {line.strip()[:120]}")
                            if len(matches) >= 5:
                                break
                except Exception:
                    continue
                
                if len(matches) >= 5:
                    break
        except Exception:
            pass
        
        return matches
