from fastapi import APIRouter
from apps.api.v1.routes import llm

api_router = APIRouter()
api_router.include_router(llm.router)

from fastapi import APIRouter
from apps.api.v1.routes.llm import router as llm_router
from apps.api.v1.routes.conversations import router as conversations_router

api_router = APIRouter()
api_router.include_router(llm_router)
api_router.include_router(conversations_router)