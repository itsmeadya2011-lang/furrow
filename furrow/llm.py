from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
import structlog
from anthropic import AsyncAnthropic, APIStatusError
from openai import AsyncOpenAI, APIStatusError as OpenAIAPIStatusError
from tenacity import retry, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self.logger = structlog.get_logger()

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = (
                os.getenv("FURROW_ANTHROPIC_API_KEY") or self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            )
            if not api_key:
                raise ValueError(
                    "Anthropic API key is not set. Checked env vars: FURROW_ANTHROPIC_API_KEY, ANTHROPIC_API_KEY"
                )
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = os.getenv("FURROW_OPENAI_API_KEY") or self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key is not set. Checked env vars: FURROW_OPENAI_API_KEY, OPENAI_API_KEY"
                )
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Complete a single prompt with the configured provider.

        Routes to the appropriate provider's completion method. Maintains backward
        compatibility with the simple prompt/response interface.

        Args:
            prompt: The user prompt to complete.
            system: Optional system message. Defaults to a helpful coding assistant.
            model: Optional model override. Defaults to the configured model.

        Returns:
            The completion response as a string.
        """
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def chat(self, messages: list[dict], model: str | None = None, **kwargs: Any) -> str:
        """Chat using a list of message dicts with the configured provider.

        Provides a higher-level interface supporting multi-turn conversations.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Optional model override. Defaults to the configured model.
            **kwargs: Additional provider-specific keyword arguments.

        Returns:
            The assistant's response as a string.
        """
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            system = kwargs.get("system", "")
            user_messages = [m for m in messages if m["role"] != "system"]
            system = " ".join(m["content"] for m in messages if m["role"] == "system") or system or "You are a helpful coding assistant."
            response = await self._complete_anthropic(
                "\n".join(m["content"] for m in user_messages if m["role"] == "user"),
                system,
                model,
            )
            return response
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai_chat(messages, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama_chat(messages, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        """Complete a prompt using Anthropic's Messages API with automatic retry.

        Retries up to 3 times on APIStatusError and ConnectionError with
        exponential backoff (2s, 4s, 8s, up to max 10s wait).

        Args:
            prompt: The user prompt.
            system: The system message.
            model: The model identifier.

        Returns:
            The assistant's text response.
        """
        self.logger.info(
            "llm.request",
            provider="anthropic",
            model=model,
            prompt_length=len(prompt),
        )
        try:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
                timeout=60.0,
            )
            self.logger.info("llm.response", provider="anthropic", model=model, status="success")
            return response.content[0].text
        except APIStatusError as e:
            self.logger.error("llm.error", provider="anthropic", model=model, status=e.status_code, error=str(e))
            raise
        except ConnectionError as e:
            self.logger.error("llm.error", provider="anthropic", model=model, error="connection_error")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        """Complete a prompt using OpenAI's Chat Completions API with automatic retry.

        Retries up to 3 times on APIStatusError and ConnectionError with
        exponential backoff (2s, 4s, 8s, up to max 10s wait).

        Args:
            prompt: The user prompt.
            system: The system message.
            model: The model identifier.

        Returns:
            The assistant's text response.
        """
        return await self._complete_openai_chat(
            [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            model,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    async def _complete_openai_chat(self, messages: list[dict], model: str) -> str:
        self.logger.info(
            "llm.request",
            provider="openai",
            model=model,
            prompt_length=sum(len(m.get("content", "")) for m in messages),
        )
        try:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages,
                timeout=60.0,
            )
            content = response.choices[0].message.content or ""
            self.logger.info("llm.response", provider="openai", model=model, status="success")
            return content
        except OpenAIAPIStatusError as e:
            self.logger.error("llm.error", provider="openai", model=model, status=e.status_code, error=str(e))
            raise
        except ConnectionError as e:
            self.logger.error("llm.error", provider="openai", model=model, error="connection_error")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        """Complete a prompt using Ollama's local API with automatic retry.

        Sends a POST request to the Ollama chat endpoint and retries on
        connection errors with exponential backoff.

        Args:
            prompt: The user prompt.
            system: The system message.
            model: The model identifier.

        Returns:
            The assistant's text response.
        """
        return await self._complete_ollama_chat(
            [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            model,
        )

    async def _complete_ollama_chat(self, messages: list[dict], model: str) -> str:
        self.logger.info(
            "llm.request",
            provider="ollama",
            model=model,
            prompt_length=sum(len(m.get("content", "")) for m in messages),
        )
        base_url = self.settings.ollama_base_url.rstrip("/")
        url = f"{base_url}/api/chat"
        payload = {"model": model, "messages": messages, "stream": False}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=60.0)
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", "")
                self.logger.info("llm.response", provider="ollama", model=model, status="success")
                return content
        except httpx.HTTPStatusError as e:
            self.logger.error("llm.error", provider="ollama", model=model, status=e.response.status_code, error=str(e))
            raise ValueError(f"Ollama API returned error: {e.response.status_code} - {e.response.text}") from e
        except httpx.ConnectionError as e:
            self.logger.error("llm.error", provider="ollama", model=model, error="connection_error")
            raise ValueError(f"Could not connect to Ollama at {url}. Is the Ollama server running?") from e
        except Exception as e:
            self.logger.error("llm.error", provider="ollama", model=model, error=str(e))
            raise

    async def read_file(self, path: str | Path) -> str:
        """Read the contents of a file asynchronously.

        Args:
            path: Path to the file to read.

        Returns:
            File contents as a string.
        """
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        """Write content to a file asynchronously, creating directories as needed.

        Args:
            path: Target file path.
            content: Content to write.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        """List all files in a directory recursively.

        Args:
            directory: Root directory to search.

        Returns:
            List of relative file paths.
        """
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
