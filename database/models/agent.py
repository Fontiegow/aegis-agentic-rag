import uuid
from typing import Optional
from sqlalchemy import String, ForeignKey, Float
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base

class AgentRun(Base):
    __tablename__ = "agent_runs"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(
        String(50), default="running"
    )  # running, completed, failed
    total_duration_sec: Mapped[Optional[float]] = mapped_column(Float)