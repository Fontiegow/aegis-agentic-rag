from fastapi import APIRouter
from apps.api.v1.routes import llm

api_router = APIRouter()
api_router.include_router(llm.router)