from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from furrow.config import Provider, Settings, settings


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""
    pass


class LLMAPIError(LLMError):
    """API error from provider."""
    pass


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama_client: httpx.AsyncClient | None = None
    
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
    def ollama(self) -> httpx.AsyncClient:
        if self._ollama_client is None:
            self._ollama_client = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url,
                timeout=60.0,
            )
        return self._ollama_client
    
    async def close(self) -> None:
        """Close any open connections."""
        if self._ollama_client:
            await self._ollama_client.aclose()
            self._ollama_client = None
    
    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> str:
        """Complete a prompt with the configured provider."""
        model = model or self.settings.model
        
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, max_tokens, temperature)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, max_tokens, temperature)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((LLMRateLimitError, LLMAPIError)),
        reraise=True,
    )
    async def _complete_anthropic(
        self,
        prompt: str,
        system: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        try:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic rate limit: {e}") from e
        except anthropic.APIError as e:
            raise LLMAPIError(f"Anthropic API error: {e}") from e
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((LLMRateLimitError, LLMAPIError)),
        reraise=True,
    )
    async def _complete_openai(
        self,
        prompt: str,
        system: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        try:
            response = await self.openai.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit: {e}") from e
        except openai.APIError as e:
            raise LLMAPIError(f"OpenAI API error: {e}") from e
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((LLMRateLimitError, LLMAPIError)),
        reraise=True,
    )
    async def _complete_ollama(
        self,
        prompt: str,
        system: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.ollama.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise LLMRateLimitError(f"Ollama rate limit: {e}") from e
            raise LLMAPIError(f"Ollama API error: {e}") from e
        except httpx.RequestError as e:
            raise LLMAPIError(f"Ollama request error: {e}") from e
    
    async def read_file(self, path: str | Path) -> str:
        """Read a file from the workspace."""
        full_path = self.settings.workspace / path if not Path(path).is_absolute() else Path(path)
        try:
            async with aiofiles.open(full_path, "r") as f:
                return await f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {full_path}")
        except Exception as e:
            raise LLMError(f"Error reading file {full_path}: {e}") from e
    
    async def write_file(self, path: str | Path, content: str) -> None:
        """Write a file to the workspace."""
        full_path = self.settings.workspace / path if not Path(path).is_absolute() else Path(path)
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(full_path, "w") as f:
                await f.write(content)
        except Exception as e:
            raise LLMError(f"Error writing file {full_path}: {e}") from e
    
    def list_files(self, directory: str | Path) -> list[str]:
        """List files in a directory relative to workspace."""
        full_path = self.settings.workspace / directory if not Path(directory).is_absolute() else Path(directory)
        if not full_path.exists():
            return []
        return [str(f.relative_to(self.settings.workspace)) for f in full_path.rglob("*") if f.is_file()]