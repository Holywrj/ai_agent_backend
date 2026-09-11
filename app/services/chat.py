from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import create_llm
from app.exceptions.base import BusinessException
from app.models.conversation import Conversation
from app.services.memory import create_conversation, get_messages, save_langchain_message
from app.tools.registry import get_all_tools


async def chat(
        db: AsyncSession,
        message: str,
        user_id: int,
        conversation_id: int | None
) -> tuple[int, str]:
    # 1. 没有 conversation_id, 创建新的 Conversation
    if conversation_id is None:
        conversation = await create_conversation(
            db=db,
            user_id=user_id
        )
    else:
        # 2. 有 conversation_id, 确认这个 Conversation 属于当前用户
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            )
        )
        conversation = result.scalar_one_or_none()
        if conversation is None:
            raise BusinessException(
                message='Conversation not found',
                code='CONVERSATION_NOT_FOUND'
            )
    # 3. 读取历史消息
    history = await get_messages(
        db=db,
        conversation_id=conversation.id
    )
    history_count = len(history)
    # 4. 历史消息已经是 LangChain Message
    messages = history.copy()
    # 5. 加入当前用户消息
    messages.append(
        HumanMessage(
            content=message
        )
    )
    # 6. 创建 LangChain Agent
    llm = create_llm()
    agent = create_agent(
        model=llm,
        tools=get_all_tools(),
        system_prompt=(
            '你是一个专业的 AI 助手。'
            '回答用户问题时要准确、简洁。'
            '如果需要查询实时天气，可以使用天气工具。'
        )
    )
    # 7. 调用 Agent
    result = await agent.ainvoke({
        'messages': messages
    })
    answer = result['messages'][-1].content
    # 8. 保存消息
    new_messages = result['messages'][history_count:]
    for msg in new_messages:
        await save_langchain_message(
            db=db,
            conversation_id=conversation.id,
            message=msg
        )

    return conversation.id, answer
