from __future__ import annotations

from pathlib import Path

import pytest

from furrow.config import Settings


def make_settings(tmp_path, max_cycles=0, max_parallel_tasks=5, max_task_retries=1,
                  max_worker_iterations=12, command_timeout=300.0, test_timeout=600.0,
                  **overrides):
    kwargs = dict(
        _env_file=None,
        workspace=tmp_path,
        max_cycles=max_cycles,
        max_parallel_tasks=max_parallel_tasks,
        max_task_retries=max_task_retries,
        max_worker_iterations=max_worker_iterations,
        command_timeout=command_timeout,
        test_timeout=test_timeout,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


class FakeLLMClient:
    def __init__(self, responses=None, response="", raises=None):
        self.raises = raises
        self.response = response
        if responses is None:
            self.responses = []
        else:
            self.responses = list(responses)
        self.calls = []
        self.prompts = []
        self.models = []

    async def chat(self, messages, tools=None, system="", model=None):
        self.calls.append({"messages": messages, "tools": tools, "system": system, "model": model})
        if self.responses:
            item = self.responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if self.raises is not None:
            raise self.raises if isinstance(self.raises, BaseException) else self.raises()
        return self.response

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        self.models.append(model)
        if self.raises is not None:
            raise self.raises if isinstance(self.raises, BaseException) else self.raises()
        return self.response

    async def read_file(self, path):
        return Path(path).read_text()

    async def write_file(self, path, content):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)

    def list_files(self, directory):
        return [str(p) for p in Path(directory).rglob("*") if p.is_file()]


class LoopingLLMClient(FakeLLMClient):
    def __init__(self, response):
        super().__init__(response=response)
        self.response = response

    async def chat(self, messages, tools=None, system="", model=None):
        self.calls.append({"messages": messages, "tools": tools, "system": system, "model": model})
        return self.response


@pytest.fixture(name="make_settings")
def _make_settings_fixture(tmp_path):
    def _make(**kwargs):
        return make_settings(tmp_path, **kwargs)
    return _make
