from __future__ import annotations

from typing import Any

import structlog

from furrow.config import Settings
from furrow.llm import LLMClient

logger = structlog.get_logger()


class ContextCompactor:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.max_context_chars = 50000  # Roughly 12k tokens

    def should_compact(self, context: str) -> bool:
        return len(context) > self.max_context_chars

    async def compact(self, context: str, goal: str) -> str:
        """Summarize context while preserving key information for the goal."""
        logger.info("context_compact_start", context_length=len(context), goal=goal[:100])

        prompt = f"""You are a context compactor for an autonomous coding agent. 
Summarize the following context while preserving all information relevant to the goal.

Goal: {goal}

Context to summarize:
{context}

Return a concise summary that includes:
1. Key decisions made
2. Files modified and their purpose
3. Current state of the implementation
4. Any blockers or issues
5. What remains to be done

Keep it under 5000 characters. Be specific and technical."""

        try:
            summary = await self.client.complete(
                prompt,
                system="You are a technical summarizer. Be concise and preserve all actionable details.",
                model=self.client.settings.model
            )
            logger.info("context_compact_done", original_length=len(context), summary_length=len(summary))
            return summary
        except (RuntimeError, ValueError, OSError) as e:
            logger.error("context_compact_failed", error=str(e))
            # Fallback: truncate
            return context[-self.max_context_chars:]

    async def compact_state(self, state: Any, goal: str) -> str:
        """Compact a FurrowState into a summary."""
        # Build context from state
        parts = [f"Goal: {goal}\n"]
        
        if hasattr(state, 'cycles') and state.cycles:
            parts.append(f"Total cycles: {len(state.cycles)}\n")
            # Include last 3 cycles in detail
            for cycle in state.cycles[-3:]:
                parts.append(f"\nCycle {cycle.cycle} ({cycle.timestamp}):")
                parts.append(f"  Rationale: {cycle.plan_rationale}")
                for task in cycle.tasks:
                    status = task.get('status', 'unknown')
                    desc = task.get('description', '')
                    parts.append(f"  Task {task.get('id')}: [{status}] {desc}")
                    if task.get('result'):
                        parts.append(f"    Result: {task.get('result')[:200]}")
                if cycle.test_result:
                    tr = cycle.test_result
                    parts.append(f"  Tests: {'PASSED' if tr.get('passed') else 'FAILED'} - {tr.get('summary')}")
                    if tr.get('failures'):
                        for f in tr.get('failures', []):
                            parts.append(f"    Failure: {f}")
        
        context = "\n".join(parts)
        
        if self.should_compact(context):
            return await self.compact(context, goal)
        return context