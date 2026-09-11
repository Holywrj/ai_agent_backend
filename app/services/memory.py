from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.models.message import Message


async def create_conversation(
        db: AsyncSession,
        user_id: int
) -> Conversation:
    conversation = Conversation(
        user_id=user_id
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    return conversation


async def add_message(
        db: AsyncSession,
        conversation_id: int,
        role: str,
        content: str
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)

    return message


async def get_messages(
        db: AsyncSession,
        conversation_id: int
) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at, Message.id)
    )

    return list(result.scalars().all())
