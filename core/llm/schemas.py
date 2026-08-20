from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None

class ChatRequest(BaseModel):
    provider: Literal["ollama", "vllm", "openrouter"] = Field(default="ollama")
    model: str = Field(..., description="The specific model ID to route to.")
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False)
    max_tokens: Optional[int] = None

class ChatResponse(BaseModel):
    id: str
    model: str
    message: Message
    usage: Optional[Dict[str, int]] = Field(
        default=None, 
        description="Token counting: prompt_tokens, completion_tokens, total_tokens"
    )