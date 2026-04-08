import asyncio
import logging

import httpx

from ..config.settings import VLMSettings
from ..utils.image_utils import image_to_base64

logger = logging.getLogger(__name__)


class AsyncVLMClient:
    def __init__(self, settings: VLMSettings):
        self._settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
            proxy=None,
        )
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def call(
        self,
        image,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        system_message: str | None = None,
        image_b64: str | None = None,
    ) -> str:
        b64 = image_b64 or image_to_base64(image)
        data_url = f"data:image/jpeg;base64,{b64}"

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        })

        return await self._send(messages, max_tokens, temperature)

    async def call_multi_image(
        self,
        images: list,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        system_message: str | None = None,
        images_b64: list[str] | None = None,
    ) -> str:
        content = []
        for i, img in enumerate(images):
            b64 = images_b64[i] if images_b64 else image_to_base64(img)
            data_url = f"data:image/jpeg;base64,{b64}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        content.append({"type": "text", "text": prompt})

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": content})

        return await self._send(messages, max_tokens, temperature)

    async def _send(
        self, messages: list, max_tokens: int, temperature: float
    ) -> str:
        payload = {
            "model": self._settings.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": self._settings.top_p,
        }

        async with self._semaphore:
            response = await self._client.post(
                self._settings.endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._settings.api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(
                self._settings.endpoint.replace("/chat/completions", "/models"),
                headers={"Authorization": f"Bearer {self._settings.api_key}"},
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self._client.aclose()
