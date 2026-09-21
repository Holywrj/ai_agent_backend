from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ticket import Ticket


async def create_support_ticket(
        db: AsyncSession,
        user_id: int,
        conversation_id: int,
        draft: dict[str, Any]
) -> Ticket:
    ticket = Ticket(
        user_id=user_id,
        conversation_id=conversation_id,
        title=draft['title'],
        description=draft['description'],
        category=draft['category'],
        priority=draft['priority'],
        status='open'
    )
    db.add(ticket)
    await db.commit()
    await db.refresh(ticket)

    return ticket
