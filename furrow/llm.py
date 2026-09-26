from __future__ import annotations

import os
from pathlib import Path

import aiofiles
import httpx
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger()


def parse_json_response(response: str) -> str:
    """Extract JSON from an LLM response that may be wrapped in markdown.

    Strips markdown code block fences and extracts the JSON portion
    if the response contains prose around the JSON.
    """
    text = response.strip()
    # Strip markdown code block fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove the opening fence (may include a language tag like ```json)
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # Remove the closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # If there is still prose around the JSON, extract the JSON object
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace : last_brace + 1]
    return text


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: httpx.AsyncClient | None = None

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
    def httpx_client(self) -> httpx.AsyncClient:
        if self._httpx is None:
            self._httpx = httpx.AsyncClient(timeout=60.0)
        return self._httpx

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        logger.debug("llm_request", provider="anthropic", model=model)
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        logger.debug("llm_request", provider="openai", model=model)
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        logger.debug(
            "llm_request", provider="ollama", model=model, base_url=self.settings.ollama_base_url
        )
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "system": system or "You are a helpful coding assistant.",
            "stream": False,
        }
        resp = await self.httpx_client.post(
            f"{self.settings.ollama_base_url}/api/generate", json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    async def read_file(self, path: str | Path) -> str:
        try:
            async with aiofiles.open(path, "r") as f:
                return await f.read()
        except FileNotFoundError:
            logger.error("file_not_found", path=str(path))
            raise
        except Exception:
            logger.exception("read_file_error", path=str(path))
            raise

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return sorted(str(f.relative_to(p)) for f in p.rglob("*") if f.is_file())
