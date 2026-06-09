"""
Shared async DeepSeek API client for all agents.

Provides a unified interface for LLM chat completion with:
- Standard chat() method for text generation
- chat_json() method for structured JSON output
- Configurable model, temperature, max_tokens per call
- Proper error handling and logging
"""

import json
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp

logger = logging.getLogger("agent.llm_client")


class LLMClient:
    """Async DeepSeek API client shared across all agents."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        system_prompt: str,
        user_input: str = "",
        context: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a chat completion request and return the response text.

        Args:
            system_prompt: System prompt for the LLM
            user_input: Current user input text
            context: Optional additional context (formatted history, etc.)
            messages: Optional pre-built message list (overrides system_prompt/user_input/context)
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Generated response text
        """
        if messages is None:
            messages = self._build_messages(system_prompt, user_input, context)

        payload = self._build_payload(messages, temperature, max_tokens)
        data = await self._post(payload)
        return self._extract_text(data)

    async def chat_json(
        self,
        system_prompt: str,
        user_input: str = "",
        context: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion request and parse the response as JSON.

        Args:
            Same as chat()

        Returns:
            Parsed JSON dict

        Raises:
            ValueError: If response is not valid JSON
        """
        if messages is None:
            messages = self._build_messages(system_prompt, user_input, context)

        payload = self._build_payload(messages, temperature, max_tokens)
        data = await self._post(payload)
        return self._extract_json(data)

    async def stream_chat(
        self,
        system_prompt: str,
        user_input: str = "",
        context: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens from the DeepSeek API.

        Uses SSE streaming (stream: true). Yields content tokens as they
        arrive, one at a time. Finishes when the [DONE] sentinel is received
        or the connection closes.

        Args:
            Same as chat()
        """
        if messages is None:
            messages = self._build_messages(system_prompt, user_input, context)

        payload = self._build_payload(messages, temperature, max_tokens)
        payload["stream"] = True

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"DeepSeek API error {response.status}: {error_text}")
                    raise RuntimeError(
                        f"DeepSeek API returned {response.status}: {error_text}"
                    )

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        return

                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        system_prompt: str,
        user_input: str,
        context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build the messages array for the API request."""
        messages = [{"role": "system", "content": system_prompt}]

        if context:
            messages.append({"role": "user", "content": context})

        if user_input:
            messages.append({"role": "user", "content": user_input})

        return messages

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build the request payload."""
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the HTTP POST to the DeepSeek API."""
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"DeepSeek API error {response.status}: {error_text}")
                    raise RuntimeError(
                        f"DeepSeek API returned {response.status}: {error_text}"
                    )
                return await response.json()

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _extract_text(self, data: Dict[str, Any]) -> str:
        """Extract text content from API response."""
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected API response structure: {e}")
            raise ValueError(f"Failed to extract text from response: {e}")

    def _extract_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and parse JSON from API response.

        Tries direct JSON parse first, then attempts to find JSON
        within markdown code blocks.
        """
        text = self._extract_text(data)

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json ... ``` block
        if "```json" in text:
            try:
                json_str = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            except (IndexError, json.JSONDecodeError):
                pass

        # Try extracting from ``` ... ``` block
        if "```" in text:
            try:
                json_str = text.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            except (IndexError, json.JSONDecodeError):
                pass

        logger.error(f"Failed to parse JSON from response: {text[:200]}")
        raise ValueError("Response is not valid JSON")


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------


def create_llm_client() -> Optional[LLMClient]:
    """Create an LLMClient from environment variables.

    Reads:
        DEEPSEEK_API_KEY (required)
        DEEPSEEK_API_BASE (default: https://api.deepseek.com)
        CHAT_MODEL (default: deepseek-chat)
        CHAT_TEMPERATURE (default: 0.7)
        CHAT_MAX_TOKENS (default: 2000)

    Returns:
        LLMClient instance, or None if DEEPSEEK_API_KEY is not set
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning("DEEPSEEK_API_KEY not set, LLMClient disabled")
        return None

    return LLMClient(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        model=os.getenv("CHAT_MODEL", "deepseek-chat"),
        temperature=float(os.getenv("CHAT_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("CHAT_MAX_TOKENS", "2000")),
    )
