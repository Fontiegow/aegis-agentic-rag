import json
import time
from typing import AsyncGenerator
import httpx

from core.config import settings
from .base import BaseLLMProvider
from .schemas import ChatRequest, ChatResponse, Message


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

    async def generate(self, request: ChatRequest) -> ChatResponse:
        payload = {
            "model": request.model,
            "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
            },
        }
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens

        response = await self.client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return ChatResponse(
            id=f"ollama-{int(time.time())}",
            model=request.model,
            message=Message(
                role=data["message"]["role"],
                content=data["message"]["content"],
            ),
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

    async def stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        payload = {
            "model": request.model,
            "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
            "stream": True,
            "options": {
                "temperature": request.temperature,
            },
        }
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens

        async with self.client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]