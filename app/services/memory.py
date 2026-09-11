from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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
        content: str,
        tool_call_id: str | None = None,
        tool_calls: list | None = None
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_call_id=tool_call_id,
        tool_calls=tool_calls
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)

    return message


async def save_langchain_message(
        db: AsyncSession,
        conversation_id: int,
        message: HumanMessage | AIMessage | ToolMessage
) -> Message:
    if isinstance(message, HumanMessage):
        role = 'user'
        content = message.content
        tool_call_id = None
        tool_calls = None
    elif isinstance(message, AIMessage):
        role = 'assistant'
        content = message.content
        tool_call_id = None
        tool_calls = message.tool_calls
    elif isinstance(message, ToolMessage):
        role = 'tool'
        content = message.content
        tool_call_id = message.tool_call_id
        tool_calls = None
    else:
        raise ValueError(
            f'Unsupported message type: {type(message)}'
        )

    return await add_message(
        db=db,
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_call_id=tool_call_id,
        tool_calls=tool_calls
    )


def to_langchain_message(
        message: Message
) -> HumanMessage | AIMessage | ToolMessage:
    if message.role == 'user':
        return HumanMessage(
            content=message.content
        )
    elif message.role == 'assistant':
        return AIMessage(
            content=message.content,
            tool_calls=message.tool_calls or []
        )
    elif message.role == 'tool':
        return ToolMessage(
            content=message.content,
            tool_call_id=message.tool_call_id
        )
    else:
        raise ValueError(
            f'Unsupported message role: {message.role}'
        )


async def get_messages(
        db: AsyncSession,
        conversation_id: int
) -> list[HumanMessage | AIMessage | ToolMessage]:
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at, Message.id)
    )

    messages = result.scalars().all()

    return [
        to_langchain_message(message)
        for message in messages
    ]
