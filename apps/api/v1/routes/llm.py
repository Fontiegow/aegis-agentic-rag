import json
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from pydantic import BaseModel, Field
from core.llm.parser import StructuredOutputParser

from core.llm.factory import get_llm_provider
from core.llm.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["LLM Gateway"])


@router.post("/completions", response_model=ChatResponse | None)
async def create_chat_completion(request: ChatRequest):
    """
    Unified chat completion endpoint supporting non-streaming responses 
    and Server-Sent Events (SSE) streaming.
    """
    try:
        provider = get_llm_provider(request.provider)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Handle Server-Sent Events (SSE) Streaming
    if request.stream:
        async def sse_stream_generator():
            try:
                async for chunk in provider.stream(request):
                    payload = json.dumps({"content": chunk})
                    yield f"data: {payload}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as err:
                error_payload = json.dumps({"error": str(err)})
                yield f"data: {error_payload}\n\n"

        return StreamingResponse(
            sse_stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Handle Non-Streaming Standard JSON Response
    try:
        return await provider.generate(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM Provider execution error: {str(e)}",
        )


# Example structured model target
class ExtractionTestSchema(BaseModel):
    summary: str = Field(..., description="Short summary of the input.")
    key_entities: list[str] = Field(..., description="Entities extracted from text.")
    sentiment: str = Field(..., description="Overall sentiment (Positive/Neutral/Negative).")


@router.post("/structured-test", response_model=ExtractionTestSchema)
async def test_structured_output(request: ChatRequest):
    """
    Test route validating forced Pydantic JSON extraction.
    """
    try:
        provider = get_llm_provider(request.provider)
        parsed_data, _ = await StructuredOutputParser.parse_response(
            provider=provider,
            request=request,
            response_model=ExtractionTestSchema
        )
        return parsed_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Structured output parsing failed: {str(e)}"
        )