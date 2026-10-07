from __future__ import annotations

import re

FENCED_BLOCK_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<fence>[`~]{3,})[ \t]*(?P<info>[^\n`~]*)\n"
    r"(?P<content>.*?)^[ \t]*(?P=fence)[ \t]*$",
    re.DOTALL | re.MULTILINE,
)

_PATH_WITH_EXTENSION_RE = re.compile(r"^[\w\-.]+\.[A-Za-z0-9]+$")


def _looks_like_path(identifier: str) -> bool:
    ident = identifier.strip()
    if not ident:
        return False
    if "/" in ident or "\\" in ident:
        return True
    if ident.startswith("."):
        return True
    return _PATH_WITH_EXTENSION_RE.fullmatch(ident) is not None


def parse_file_edits(response: str) -> list[tuple[str, str]]:
    """Parse file edits from an LLM response.

    File edits are fenced code blocks whose info string is a file path:

        ```src/app/main.py
        print("hello")
        ```

    Returns a list of ``(path, content)`` tuples in order of appearance.
    Blocks tagged with a programming language (e.g. ```python) or with no
    info string are ignored, so an empty list means "no file edits".
    """
    edits: list[tuple[str, str]] = []
    for match in FENCED_BLOCK_RE.finditer(response):
        path = match.group("info").strip()
        if not _looks_like_path(path):
            continue
        content = match.group("content")
        indent = match.group("indent")
        if indent:
            content = "\n".join(
                line.removeprefix(indent)
                for line in content.split("\n")
            )
        edits.append((path, content.strip("\n")))
    return edits
