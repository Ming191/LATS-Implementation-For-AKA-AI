"""LLM client for DeepSeek integration with retry logic."""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import httpx
from core.config import settings


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    api_key: str
    base_url: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60


class DeepSeekClient:
    """
    DeepSeek API client with exponential backoff retry logic.

    Handles:
    - Async HTTP calls to DeepSeek API
    - Exponential backoff retry (3 attempts: 2s, 4s, 8s)
    - JSON response parsing
    - Error handling for rate limits, timeouts, invalid responses
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = LLMConfig(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                model=settings.deepseek_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout=60,
            )
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout),
            headers={"Authorization": f"Bearer {config.api_key}"},
        )

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None,
    ) -> str:
        """
        Generate completion from DeepSeek with retry logic.

        Args:
            prompt: User prompt
            temperature: Override default temperature (0.0-2.0)
            max_tokens: Override default max tokens
            system_message: Optional system message

        Returns:
            Generated text content

        Raises:
            Exception: If all retry attempts fail
        """
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Retry with exponential backoff: 2s, 4s, 8s
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    f"{self.config.base_url}/chat/completions", json=payload
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    return content
                elif response.status_code == 429:
                    # Rate limit - wait and retry
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    error_msg = f"API error: {response.status_code} - {response.text}"
                    if attempt == max_retries - 1:
                        raise Exception(error_msg)
                    await asyncio.sleep(2**attempt)

            except httpx.TimeoutException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Request timeout after {max_retries} attempts") from e
                await asyncio.sleep(2**attempt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"LLM generation failed: {str(e)}") from e
                await asyncio.sleep(2**attempt)

        raise Exception(f"Failed to generate after {max_retries} attempts")

    async def generate_batch(
        self, prompts: List[str], temperature: float = 0.7, max_tokens: Optional[int] = None
    ) -> List[str]:
        """
        Generate multiple completions in parallel.

        Args:
            prompts: List of user prompts
            temperature: Temperature for all prompts
            max_tokens: Max tokens for all prompts

        Returns:
            List of generated texts (same order as prompts)
        """
        tasks = [self.generate(prompt, temperature, max_tokens) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error strings
        outputs = []
        for result in results:
            if isinstance(result, Exception):
                outputs.append(f"ERROR: {str(result)}")
            else:
                outputs.append(result)

        return outputs

    def extract_json_from_response(self, response: str) -> Any:
        """
        Extract JSON from LLM response, handling markdown code blocks.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed JSON object (dict or list)

        Raises:
            ValueError: If no valid JSON found
        """
        # Try direct JSON parse first
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        # Pattern: ```json\n...\n``` or ```\n...\n```
        patterns = [
            r"```json\s*\n(.*?)\n```",
            r"```\s*\n(.*?)\n```",
            r"\[.*?\]",  # Match array anywhere
            r"\{.*?\}",  # Match object anywhere
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1) if match.lastindex else match.group(0)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"Could not extract valid JSON from response: {response[:200]}...")

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
