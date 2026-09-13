from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.exceptions.base import BusinessException
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
        conversation_id: int,
        limit: int = 50
) -> list[HumanMessage | AIMessage | ToolMessage]:
    """
    获取最近的历史消息，并保证从完整的用户轮次开始。
    :param db: db session
    :param conversation_id: 会话 id
    :param limit: 初始消息数量限制，如果limit刚好切到了某一轮对话中间，会向前补齐，该轮到user消息，最终返回数量可能超过limit
    :return: list[HumanMessage | AIMessage | ToolMessage]
    """
    if limit <= 0:
        raise BusinessException(
            message='limit must be greater than 0',
            code='LIMIT_TOO_SMALL'
        )
    # 1. 先获取最近N条消息
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(
            Message.created_at.desc(),
            Message.id.desc()
        )
        .limit(limit)
    )
    recent_messages = list(result.scalars().all())
    if not recent_messages:
        return []
    # 2. 恢复正常的时间顺序
    recent_messages.reverse()
    # 3. 如果切到了某一轮中间，则向前补齐到user
    if recent_messages[0].role != 'user':
        first_message_id = recent_messages[0].id
        # 4. 找到这条消息之前最近的user消息
        result = await db.execute(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.id < first_message_id,
                Message.role == 'user'
            )
            .order_by(Message.id.desc())
            .limit(1)
        )
        first_user_message = result.scalar_one_or_none()
        # 5. 从当前的user消息开始读取
        if first_user_message is not None:
            result = await db.execute(
                select(Message)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.id >= first_user_message.id
                )
                .order_by(Message.created_at, Message.id)
            )
            recent_messages = list(result.scalars().all())

    # 6. 数据库 Message -> LangChain Message
    return [
        to_langchain_message(message)
        for message in recent_messages
    ]
