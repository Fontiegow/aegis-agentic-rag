from abc import ABC, abstractmethod
from typing import AsyncGenerator
from .schemas import ChatRequest, ChatResponse

class BaseLLMProvider(ABC):
    """
    Abstract interface for all LLM providers.
    """
    
    @abstractmethod
    async def generate(self, request: ChatRequest) -> ChatResponse:
        """
        Process a non-streaming chat completion request.
        """
        pass

    @abstractmethod
    async def stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """
        Process a streaming chat completion request via Server-Sent Events (SSE).
        """
        pass