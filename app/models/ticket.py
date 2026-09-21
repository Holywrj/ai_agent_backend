from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Ticket(Base):
    __tablename__ = 'tickets'

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id'),
        nullable=False,
        index=True
    )
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey('conversations.id'),
        nullable=False,
        index=True
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    category: Mapped[str] = mapped_column(
        String(30),
        nullable=False
    )
    priority: Mapped[int] = mapped_column(
        String(20),
        nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default='open'
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
