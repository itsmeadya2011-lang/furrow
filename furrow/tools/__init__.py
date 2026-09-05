from __future__ import annotations

from furrow.tools.base import Tool, ToolResult
from furrow.tools.file_ops import EditFile, GrepFiles, ListFiles, ReadFile, WriteFile
from furrow.tools.execute import ExecuteCommand

__all__ = [
    "Tool",
    "ToolResult",
    "ReadFile",
    "WriteFile",
    "EditFile",
    "ListFiles",
    "GrepFiles",
    "ExecuteCommand",
]

# Default toolset for coding agents
DEFAULT_TOOLS: list[type[Tool]] = [
    ReadFile,
    WriteFile,
    EditFile,
    ListFiles,
    GrepFiles,
    ExecuteCommand,
]
