from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result of executing a tool."""

    success: bool
    output: str
    error: str | None = None


class Tool(ABC):
    """Base class for all tools available to agentic LLM calls."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The tool name as seen by the LLM."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for the tool's input parameters (Anthropic format)."""
        return {
            "type": "object",
            "properties": self._properties,
            "required": self._required,
        }

    @property
    def _properties(self) -> dict[str, Any]:
        return {}

    @property
    def _required(self) -> list[str]:
        return []

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a result."""

    def to_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
            },
        }
        properties = self.input_schema.get("properties", {})
        required = self.input_schema.get("required", [])
        if properties:
            schema["function"]["parameters"] = {
                "type": "object",
                "properties": properties,
            }
            if required:
                schema["function"]["parameters"]["required"] = required
        else:
            schema["function"]["parameters"] = {"type": "object", "properties": {}}
        return schema


__all__ = ["Tool", "ToolResult"]
