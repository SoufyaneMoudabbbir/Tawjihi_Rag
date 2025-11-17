#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek API Client
Handles communication with DeepSeek LLM API
"""
import httpx
import json
from typing import AsyncGenerator, Optional, Dict, List
from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import DeepSeekAPIError

logger = get_logger(__name__)


class DeepSeekClient:
    """Client for DeepSeek API"""

    def __init__(self):
        self.api_key = settings.DEEPSEEK_API_KEY
        self.api_url = settings.DEEPSEEK_API_URL
        self.model = settings.DEEPSEEK_MODEL
        self.timeout = settings.DEEPSEEK_TIMEOUT

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> str:
        """
        Generate non-streaming response

        Args:
            system_prompt: System instructions
            user_prompt: User query
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            Generated text

        Raises:
            DeepSeekAPIError: If API call fails
        """
        try:
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers=self._get_headers(),
                    json=data,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                else:
                    error_msg = f"DeepSeek API error: {response.status_code}"
                    logger.error(f"{error_msg} - {response.text}")
                    raise DeepSeekAPIError(
                        error_msg,
                        response.status_code,
                        response.text
                    )

        except httpx.TimeoutException:
            logger.error("DeepSeek API timeout")
            raise DeepSeekAPIError("API request timed out", 504)
        except httpx.HTTPError as e:
            logger.error(f"DeepSeek HTTP error: {e}")
            raise DeepSeekAPIError(str(e), 500)
        except Exception as e:
            logger.error(f"DeepSeek unexpected error: {e}")
            raise DeepSeekAPIError(str(e), 500)

    async def generate_response_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response

        Args:
            system_prompt: System instructions
            user_prompt: User query
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Yields:
            Text chunks

        Raises:
            DeepSeekAPIError: If API call fails
        """
        try:
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }

            async with httpx.AsyncClient() as client:
                async with client.stream(
                    'POST',
                    self.api_url,
                    headers=self._get_headers(),
                    json=data,
                    timeout=self.timeout
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise DeepSeekAPIError(
                            f"API error: {response.status_code}",
                            response.status_code,
                            error_text.decode()
                        )

                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            json_str = line[6:]

                            if json_str == '[DONE]':
                                break

                            try:
                                chunk = json.loads(json_str)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue

        except httpx.TimeoutException:
            logger.error("DeepSeek API stream timeout")
            raise DeepSeekAPIError("API stream timed out", 504)
        except httpx.HTTPError as e:
            logger.error(f"DeepSeek stream HTTP error: {e}")
            raise DeepSeekAPIError(str(e), 500)
        except Exception as e:
            logger.error(f"DeepSeek stream unexpected error: {e}")
            raise DeepSeekAPIError(str(e), 500)
