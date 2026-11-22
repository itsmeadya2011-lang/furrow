"""Custom tool support for Furrow.

This module provides a mechanism for users to register custom tools that can be
used by the worker agent. Tools are defined as async functions that take a
workspace path and arguments, and return a string result.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CustomTool:
    """Definition of a custom tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[Path, dict[str, Any]], Awaitable[str]]
    required: list[str] = field(default_factory=list)


class ToolRegistry:
    """Registry for custom tools."""

    def __init__(self) -> None:
        self._tools: dict[str, CustomTool] = {}

    def register(self, tool: CustomTool) -> None:
        """Register a custom tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Unregister a custom tool by name."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> CustomTool | None:
        """Get a custom tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[CustomTool]:
        """List all registered custom tools."""
        return list(self._tools.values())

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions in the format expected by the LLM."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "required": tool.required,
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, workspace: Path, args: dict[str, Any]) -> str:
        """Execute a custom tool."""
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown custom tool: {name}"
        try:
            return await tool.handler(workspace, args)
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"


# Global registry instance
_registry = ToolRegistry()


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    handler: Callable[[Path, dict[str, Any]], Awaitable[str]],
    required: list[str] | None = None,
) -> None:
    """Register a custom tool globally."""
    tool = CustomTool(
        name=name,
        description=description,
        parameters=parameters,
        handler=handler,
        required=required or [],
    )
    _registry.register(tool)


def unregister_tool(name: str) -> bool:
    """Unregister a custom tool globally."""
    return _registry.unregister(name)


def get_custom_tools() -> list[dict[str, Any]]:
    """Get all custom tool definitions."""
    return _registry.get_tool_definitions()


async def execute_custom_tool(name: str, workspace: Path, args: dict[str, Any]) -> str:
    """Execute a custom tool."""
    return await _registry.execute(name, workspace, args)


def create_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    required: list[str] | None = None,
) -> Callable[
    [Callable[[Path, dict[str, Any]], Awaitable[str]]],
    Callable[[Path, dict[str, Any]], Awaitable[str]],
]:
    """Decorator to create and register a custom tool.

    Usage:
        @create_tool(
            "my_tool",
            "Description of my tool",
            {"type": "object", "properties": {"arg1": {"type": "string"}}, "required": ["arg1"]},
        )
        async def my_tool(workspace: Path, args: dict[str, Any]) -> str:
            return f"Result: {args['arg1']}"
    """
    def decorator(
        func: Callable[[Path, dict[str, Any]], Awaitable[str]],
    ) -> Callable[[Path, dict[str, Any]], Awaitable[str]]:
        register_tool(name, description, parameters, func, required or [])
        return func

    return decorator