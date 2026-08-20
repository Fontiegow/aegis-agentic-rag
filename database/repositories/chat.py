import uuid
from typing import List, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models.chat import Conversation, Message


class ChatRepository:
    """Async repository for handling conversation and message persistence."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_conversation(self, title: Optional[str] = None) -> Conversation:
        conversation = Conversation(title=title or "New Chat")
        self.session.add(conversation)
        await self.session.commit()
        # Fetch with eagerly loaded messages to prevent MissingGreenlet errors on serialization
        return await self.get_conversation(conversation.id)

    async def get_conversation(self, conversation_id: uuid.UUID) -> Optional[Conversation]:
        stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_conversations(self, limit: int = 20, offset: int = 0) -> List[Conversation]:
        stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .order_by(Conversation.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def add_message(
        self,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        meta_data: Optional[dict] = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            meta_data=meta_data or {},
        )
        self.session.add(message)
        await self.session.commit()
        await self.session.refresh(message)
        return message

    async def delete_conversation(self, conversation_id: uuid.UUID) -> bool:
        stmt = delete(Conversation).where(Conversation.id == conversation_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0