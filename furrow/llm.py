from __future__ import annotations

import json
import os
import re
from pathlib import Path

import aiofiles
import anthropic
import openai
import structlog
import tenacity
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger("furrow.llm")


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ValueError) and "is not set" in str(exc):
        return False
    return True


class JSONParseError(ValueError):
    pass


def extract_json(text: str) -> dict:
    if text is None:
        raise JSONParseError("Cannot parse JSON from None")
    cleaned = text.strip()

    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    if not cleaned:
        raise JSONParseError("Empty text, no JSON found")

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as e:
            raise JSONParseError(
                f"Failed to parse JSON: {e}\nRaw text (truncated): {cleaned[:500]!r}"
            ) from e

    raise JSONParseError(f"No JSON object found in text (truncated): {cleaned[:500]!r}")


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        if self._ollama is None:
            self._ollama = AsyncOpenAI(
                base_url=f"{self.settings.ollama_base_url}/v1",
                api_key="ollama",
            )
        return self._ollama

    def extract_json(self, text: str) -> dict:
        return extract_json(text)

    def _retry(self):
        return tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(self.settings.retry_attempts),
            wait=tenacity.wait_exponential(multiplier=1, min=1, max=30),
            retry=tenacity.retry_if_exception(_is_retryable),
            reraise=True,
        )

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_openai(prompt, system, model, client=self.ollama)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        logger.debug("anthropic.complete", model=model)
        async for attempt in self._retry():
            with attempt:
                try:
                    response = await self.anthropic.messages.create(
                        model=model,
                        max_tokens=4096,
                        timeout=self.settings.request_timeout,
                        system=system or "You are a helpful coding assistant.",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.content[0].text
                except ValueError as e:
                    if "is not set" in str(e):
                        raise
                    logger.warning("anthropic.error", error=str(e))
                    raise
        return ""

    async def _complete_openai(self, prompt: str, system: str, model: str, client: AsyncOpenAI | None = None) -> str:
        client = client or self.openai
        logger.debug("openai.complete", model=model)
        async for attempt in self._retry():
            with attempt:
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        timeout=self.settings.request_timeout,
                        messages=[
                            {"role": "system", "content": system or "You are a helpful coding assistant."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    return response.choices[0].message.content or ""
                except ValueError as e:
                    if "is not set" in str(e):
                        raise
                    logger.warning("openai.error", error=str(e))
                    raise
        return ""

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
