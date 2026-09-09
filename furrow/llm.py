from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings

logger = __import__("structlog").get_logger("furrow.llm")


class LLMClient:
    """Client for interacting with multiple LLM providers."""

    def __init__(self, settings: Settings = settings) -> None:
        """
        Initialize the LLM client with configuration.

        Args:
            settings: Application settings containing provider configuration.
        """
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        """
        Lazy-load the Anthropic client.

        Returns:
            Configured AsyncAnthropic client.

        Raises:
            ValueError: If ANTHROPIC_API_KEY is not set.
        """
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        """
        Lazy-load the OpenAI client.

        Returns:
            Configured AsyncOpenAI client.

        Raises:
            ValueError: If OPENAI_API_KEY is not set.
        """
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        """
        Lazy-load the Ollama client.

        Uses the OpenAI-compatible interface at the configured ollama_base_url.

        Returns:
            Configured AsyncOpenAI client pointed at Ollama.
        """
        if self._ollama is None:
            self._ollama = AsyncOpenAI(
                base_url=self.settings.ollama_base_url,
                api_key="ollama",
            )
        return self._ollama

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """
        Send a prompt to the configured LLM provider and return the response text.

        Args:
            prompt: The user prompt to send.
            system: Optional system prompt override.
            model: Optional model identifier override.

        Returns:
            The text response from the LLM.

        Raises:
            ValueError: If the configured provider is unsupported.
        """
        model = model or self.settings.model
        logger.info("llm_request", provider=self.settings.provider.value, model=model)
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        """
        Complete a request using the Anthropic provider.

        Args:
            prompt: The user prompt.
            system: The system prompt.
            model: The model identifier.

        Returns:
            The text response from Anthropic.
        """
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        logger.info("llm_response", provider="anthropic", model=model, length=len(text))
        return text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        """
        Complete a request using the OpenAI provider.

        Args:
            prompt: The user prompt.
            system: The system prompt.
            model: The model identifier.

        Returns:
            The text response from OpenAI.
        """
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        logger.info("llm_response", provider="openai", model=model, length=len(text))
        return text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        """
        Complete a request using the Ollama provider via OpenAI-compatible API.

        Args:
            prompt: The user prompt.
            system: The system prompt.
            model: The model identifier.

        Returns:
            The text response from Ollama.
        """
        response = await self.ollama.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        logger.info("llm_response", provider="ollama", model=model, length=len(text))
        return text

    async def read_file(self, path: str | Path) -> str:
        """
        Read the contents of a file asynchronously.

        Args:
            path: Path to the file.

        Returns:
            File contents as a string.
        """
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        """
        Write content to a file asynchronously, creating parent directories if needed.

        Args:
            path: Destination file path.
            content: Content to write.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        """
        Recursively list all files in a directory.

        Args:
            directory: Directory to scan.

        Returns:
            List of file paths relative to the directory.
        """
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
