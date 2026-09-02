from __future__ import annotations

import json
from pathlib import Path

import aiofiles


class StateManager:
    """Manages persistent state for goal/task tracking across sessions."""

    def __init__(self, workspace: Path | None = None) -> None:
        self.workspace = workspace if workspace is not None else Path.cwd() / ".furrow"
        self.state_file = self.workspace / "state.json"

    async def save(self, state: dict) -> None:
        """Save state as JSON to {workspace}/state.json, creating the directory if needed."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(self.state_file, "w") as f:
            await f.write(json.dumps(state))

    async def load(self) -> dict | None:
        """Load and return state from JSON file. Returns None if the file doesn't exist."""
        if not self.state_file.exists():
            return None
        async with aiofiles.open(self.state_file, "r") as f:
            content = await f.read()
        return json.loads(content)

    async def clear(self) -> None:
        """Remove the state file."""
        if self.state_file.exists():
            self.state_file.unlink()
