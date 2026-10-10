from __future__ import annotations

from pathlib import Path

import aiofiles


async def read_file(path: str | Path) -> str:
    async with aiofiles.open(path, "r") as f:
        return await f.read()


async def write_file(path: str | Path, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(p, "w") as f:
        await f.write(content)


async def list_files(directory: str | Path) -> list[str]:
    p = Path(directory)
    if not p.exists():
        return []
    return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
