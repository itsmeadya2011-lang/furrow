from pathlib import Path


class MockLLMClient:
    def __init__(self, response="ok", responses=None):
        self._response = response
        self._responses = list(responses) if responses else None
        self._call_count = 0
        self.settings = None

    async def complete(self, prompt="", system="", model=None):
        if self._responses is not None:
            idx = min(self._call_count, len(self._responses) - 1)
            self._call_count += 1
            return self._responses[idx]
        self._call_count += 1
        return self._response

    async def read_file(self, path):
        return Path(path).read_text()

    async def write_file(self, path, content):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def list_files(self, directory):
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
