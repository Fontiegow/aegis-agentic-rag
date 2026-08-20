from fastapi import APIRouter
from apps.api.v1.routes import llm

api_router = APIRouter()
api_router.include_router(llm.router)

from apps.api.v1.routes.llm import router as llm_router
from apps.api.v1.routes.conversations import router as conversations_router

api_router = APIRouter()
api_router.include_router(llm_router)
api_router.include_router(conversations_router)

from apps.api.v1.routes.rag import router as rag_router

api_router = APIRouter()
api_router.include_router(llm_router)
api_router.include_router(conversations_router)
api_router.include_router(rag_router) # Mount RAG router



from apps.api.v1.routes.rag import router as rag_router

api_router = APIRouter()

# Mount rag_router under /rag
api_router.include_router(rag_router, prefix="/rag", tags=["RAG"])