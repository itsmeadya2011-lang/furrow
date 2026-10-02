from __future__ import annotations

from furrow.tools.base import Tool, ToolResult
from furrow.tools.filesystem import GrepTool, ListFilesTool, ReadFileTool, WriteFileTool
from furrow.tools.shell import ShellTool


def get_default_tools() -> list[Tool]:
    """Return the standard set of tools available to workers."""
    return [
        ReadFileTool(),
        WriteFileTool(),
        ListFilesTool(),
        GrepTool(),
        ShellTool(),
    ]


__all__ = [
    "GrepTool",
    "ListFilesTool",
    "ReadFileTool",
    "ShellTool",
    "Tool",
    "ToolResult",
    "WriteFileTool",
    "get_default_tools",
]
