from langchain.agents import create_agent
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import create_llm
from app.models.conversation import Conversation
from app.services.memory import add_message, create_conversation, get_messages
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
            raise ValueError('Conversation not found')
    # 3. 读取历史消息
    history = await get_messages(
        db=db,
        conversation_id=conversation.id
    )
    # 4. 把数据库 Message 转换成 LangChain 可以理解的消息
    messages = [
        {
            'role': item.role,
            'content': item.content
        }
        for item in history
    ]
    # 5. 加入当前用户消息
    messages.append({
        'role': 'user',
        'content': message
    })
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
    # 8. 保存用户消息
    await add_message(
        db=db,
        conversation_id=conversation.id,
        role='user',
        content=message
    )
    # 9. 保存 AI 消息
    await add_message(
        db=db,
        conversation_id=conversation.id,
        role='assistant',
        content=answer
    )

    return conversation.id, answer
