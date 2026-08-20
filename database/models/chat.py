import uuid
from typing import List, Optional
from sqlalchemy import String, Text, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

class Conversation(Base):
    __tablename__ = "conversations"

    title: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Relationships
    messages: Mapped[List["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )

class Message(Base):
    __tablename__ = "messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(50))  # system, user, assistant, tool
    content: Mapped[str] = mapped_column(Text)
    meta_data: Mapped[Optional[dict]] = mapped_column(JSON)  # Token usage, tool args

    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")