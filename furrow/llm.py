from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import ollama
from anthropic import AsyncAnthropic, RateLimitError as AnthropicRateLimitError, APIError as AnthropicAPIError
from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError, APIError as OpenAIAPIError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

# Cost tracking (approximate per 1K tokens)
MODEL_COSTS = {
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022": {"input": 0.00025, "output": 0.00125},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    # OpenAI
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: ollama.AsyncClient | None = None
        self.total_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

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
    def ollama_client(self) -> ollama.AsyncClient:
        if self._ollama is None:
            self._ollama = ollama.AsyncClient(host=self.settings.ollama_base_url)
        return self._ollama

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((AnthropicRateLimitError, AnthropicAPIError, OpenAIRateLimitError, OpenAIAPIError, Exception)),
    )
    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        model = model or self.settings.model
        
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, temperature)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, temperature)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model, temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(
        self, prompt: str, system: str, model: str, temperature: float
    ) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        
        # Track usage
        if hasattr(response, 'usage'):
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            self._track_usage(model, input_tokens, output_tokens)
        
        return response.content[0].text

    async def _complete_openai(
        self, prompt: str, system: str, model: str, temperature: float
    ) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        
        # Track usage
        if hasattr(response, 'usage'):
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            self._track_usage(model, input_tokens, output_tokens)
        
        return response.choices[0].message.content or ""

    async def _complete_ollama(
        self, prompt: str, system: str, model: str, temperature: float
    ) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = await self.ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            options={"temperature": temperature},
        )
        
        # Ollama doesn't provide token counts easily, approximate
        content = response["message"]["content"]
        input_tokens = len(full_prompt) // 4
        output_tokens = len(content) // 4
        self._track_usage(model, input_tokens, output_tokens)
        
        return content

    def _track_usage(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        costs = MODEL_COSTS.get(model, {"input": 0, "output": 0})
        cost = (input_tokens / 1000) * costs["input"] + (output_tokens / 1000) * costs["output"]
        self.total_cost += cost

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

    def get_cost_summary(self) -> dict[str, Any]:
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }