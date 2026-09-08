from fastapi import APIRouter

from apps.api.v1.routes.conversations import router as conversations_router
from apps.api.v1.routes.llm import router as llm_router
from apps.api.v1.routes.rag import router as rag_router

api_router = APIRouter()

# Mount feature routers under their respective prefixes
api_router.include_router(llm_router, prefix="/llm", tags=["LLM"])
api_router.include_router(conversations_router, prefix="/conversations", tags=["Conversations"])
api_router.include_router(rag_router, prefix="/rag", tags=["RAG"])