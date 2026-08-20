import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db_session
from database.repositories.chat import ChatRepository

router = APIRouter(prefix="/conversations", tags=["Conversations Memory"])


# Schemas
class MessageRead(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    meta_data: Optional[dict] = None

    class Config:
        from_attributes = True


class ConversationRead(BaseModel):
    id: uuid.UUID
    title: Optional[str] = None
    messages: List[MessageRead] = []

    class Config:
        from_attributes = True


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(default="New Chat", max_length=255)


# Endpoints
@router.post("/", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreate,
    session: AsyncSession = Depends(get_db_session),
):
    repo = ChatRepository(session)
    return await repo.create_conversation(title=payload.title)


@router.get("/", response_model=List[ConversationRead])
async def list_conversations(
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_db_session),
):
    repo = ChatRepository(session)
    return await repo.list_conversations(limit=limit, offset=offset)


@router.get("/{conversation_id}", response_model=ConversationRead)
async def get_conversation(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
):
    repo = ChatRepository(session)
    conversation = await repo.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )
    return conversation


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
):
    repo = ChatRepository(session)
    deleted = await repo.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )