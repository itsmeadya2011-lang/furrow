from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import structlog
from anthropic import APIError as AnthropicAPIError
from anthropic import AsyncAnthropic
from openai import APIError as OpenAIAPIError
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger()

# Optional Google import
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._google_model: Any = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
            logger.debug("anthropic_client_initialized")
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
            logger.debug("openai_client_initialized")
        return self._openai

    @property
    def google(self) -> Any:
        if not GOOGLE_AVAILABLE:
            raise ValueError("Google Generative AI library not installed. Install with: pip install google-generativeai")
        if self._google_model is None:
            api_key = self.settings.google_api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not set")
            genai.configure(api_key=api_key)
            self._google_model = genai.GenerativeModel(self.settings.model)
            logger.debug("google_client_initialized")
        return self._google_model

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.info("llm_complete_request", provider=self.settings.provider.value, model=model, prompt_length=len(prompt))
        try:
            if self.settings.provider == Provider.ANTHROPIC:
                return await self._complete_anthropic(prompt, system, model)
            elif self.settings.provider == Provider.OPENAI:
                return await self._complete_openai(prompt, system, model)
            elif self.settings.provider == Provider.GOOGLE:
                return await self._complete_google(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {self.settings.provider}")
        except Exception as e:
            logger.error("llm_complete_failed", provider=self.settings.provider.value, model=model, error=str(e))
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((AnthropicAPIError, OpenAIAPIError, TimeoutError)),
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        logger.debug("anthropic_completion_start", model=model)
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.content[0].text
        logger.debug("anthropic_completion_done", model=model, response_length=len(result))
        return result

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((OpenAIAPIError, TimeoutError)),
    )
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        logger.debug("openai_completion_start", model=model)
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        result = response.choices[0].message.content or ""
        logger.debug("openai_completion_done", model=model, response_length=len(result))
        return result

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((TimeoutError, Exception)),
    )
    async def _complete_google(self, prompt: str, system: str, model: str) -> str:
        logger.debug("google_completion_start", model=model)
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = await self.google.generate_content_async(
            full_prompt,
            generation_config=GenerationConfig(
                max_output_tokens=4096,
                temperature=0.7,
            ),
        )
        result = response.text or ""
        logger.debug("google_completion_done", model=model, response_length=len(result))
        return result

    async def read_file(self, path: str | Path) -> str:
        logger.debug("read_file", path=str(path))
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
        logger.debug("read_file_done", path=str(path), length=len(content))
        return content

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        logger.debug("write_file", path=str(p), length=len(content))
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)
        logger.debug("write_file_done", path=str(p))

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        files = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
        logger.debug("list_files", directory=str(p), count=len(files))
        return files