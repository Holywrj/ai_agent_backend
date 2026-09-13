from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import create_llm
from app.exceptions.base import BusinessException
from app.models.conversation import Conversation
from app.services.context import select_messages_by_token_budget
from app.services.memory import create_conversation, get_messages, save_langchain_message
from app.services.summary import update_summary
from app.tools.registry import get_all_tools

HISTORY_MESSAGE_LIMIT = 50
HISTORY_TOKEN_BUDGET = 6000


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
    # 3. 创建 LLM
    llm = create_llm()
    # 4. 如果历史消息达到条件，先更新Summary
    await update_summary(
        db=db,
        conversation=conversation
    )
    # 5. 获取最近历史消息
    history = await get_messages(
        db=db,
        conversation_id=conversation.id,
        limit=HISTORY_MESSAGE_LIMIT
    )
    # 6. 根据 Token Budget 筛选历史
    history = select_messages_by_token_budget(
        messages=history,
        token_budget=HISTORY_TOKEN_BUDGET,
        token_counter=llm.get_num_tokens_from_messages
    )
    history_count = len(history)
    # 7. 历史消息已经是 LangChain Message
    messages = history.copy()
    # 8. 加入当前用户消息
    messages.append(
        HumanMessage(
            content=message
        )
    )
    # 9. 创建 System Prompt
    system_prompt = (
        '你是一个专业的 AI 助手。'
        '回答用户问题时要准确、简洁。'
        '如果需要查询实时天气，可以使用天气工具。'
    )
    # 10. 如果存在Summary，把它作为额外上下文
    if conversation.summary:
        messages.insert(
            0,
            SystemMessage(
                content=(
                    '以下是当前会话较早历史的摘要。'
                    '它用于帮助你理解长期上下文。'
                    '如果摘要与最近的原始消息存在冲突，'
                    '优先相信最近的原始消息。\n\n'
                    f'会话摘要：\n{conversation.summary}'
                )
            )
        )
    # 11. 创建LangChain Agent
    agent = create_agent(
        model=llm,
        tools=get_all_tools(),
        system_prompt=system_prompt
    )
    # 12. 调用 Agent
    result = await agent.ainvoke({
        'messages': messages
    })
    answer = result['messages'][-1].content
    # 13. 只保存本次请求产生的新消息
    new_messages = result['messages'][history_count + 1:]
    for msg in new_messages:
        await save_langchain_message(
            db=db,
            conversation_id=conversation.id,
            message=msg
        )
    # 14. 当前用户消息单独保存
    await save_langchain_message(
        db=db,
        conversation_id=conversation.id,
        message=HumanMessage(content=message)
    )

    return conversation.id, answer
