from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Result of executing a tool."""

    success: bool = True
    output: str = ""
    error: str | None = None


class Tool(ABC):
    """Base class for LLM-callable tools.

    Subclasses define ``name``, ``description``, and an async ``execute``
    classmethod. The ``to_schema`` methods return provider-agnostic
    schemas that can be converted to Anthropic or OpenAI formats.
    """

    name: str = ""
    description: str = ""

    @classmethod
    @abstractmethod
    async def execute(cls, **kwargs: Any) -> str | ToolResult:
        """Execute the tool and return its result."""

    @classmethod
    def _get_params(cls) -> dict[str, Any]:
        """Introspect the execute signature and build a JSON Schema
        ``parameters`` object.
        """
        sig = inspect.signature(cls.execute)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, param in sig.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                annotation = "string"
            else:
                annotation = _annotation_to_schema(param.annotation)
            properties[name] = {"type": annotation}
            if param.default is inspect.Parameter.empty:
                required.append(name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    @classmethod
    def to_openai_schema(cls) -> dict[str, Any]:
        """Return the tool definition in OpenAI tool-calling format."""
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": cls._get_params(),
            },
        }

    @classmethod
    def to_anthropic_schema(cls) -> dict[str, Any]:
        """Return the tool definition in Anthropic tool-use format."""
        return {
            "name": cls.name,
            "description": cls.description,
            "input_schema": cls._get_params(),
        }


def _annotation_to_schema(annotation: Any) -> str:
    """Map Python type annotations to JSON Schema type strings."""
    import types
    import typing

    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        args = typing.get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_schema(non_none[0])
        return "string"
    if annotation in (int,):
        return "integer"
    if annotation in (float,):
        return "number"
    if annotation in (bool,):
        return "boolean"
    if annotation in (list,):
        return "array"
    if annotation in (dict,):
        return "object"
    return "string"
