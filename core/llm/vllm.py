import json
import time
from typing import AsyncGenerator, Optional
import httpx

from core.config import settings
from .base import BaseLLMProvider
from .schemas import ChatRequest, ChatResponse, Message


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(
        self, 
        base_url: str = settings.VLLM_BASE_URL, 
        api_key: Optional[str] = None
    ):
        self.base_url = base_url.rstrip("/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        self.client = httpx.AsyncClient(
            base_url=self.base_url, 
            headers=headers, 
            timeout=60.0
        )

    async def generate(self, request: ChatRequest) -> ChatResponse:
        payload = {
            "model": request.model,
            "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
            "temperature": request.temperature,
            "stream": False,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]["message"]
        usage_data = data.get("usage", {})

        return ChatResponse(
            id=data.get("id", f"vllm-{int(time.time())}"),
            model=data.get("model", request.model),
            message=Message(
                role=choice["role"],
                content=choice["content"],
            ),
            usage={
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            },
        )

    async def stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        payload = {
            "model": request.model,
            "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
            "temperature": request.temperature,
            "stream": True,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        async with self.client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue