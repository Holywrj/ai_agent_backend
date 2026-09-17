from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import create_llm
from app.exceptions.base import BusinessException
from app.models.conversation import Conversation
from app.services.context import select_messages_by_token_budget
from app.services.memory import create_conversation, get_messages, save_langchain_message
from app.services.redis_memory import get_recent_messages, rebuild_memory, save_messages
from app.services.summary import update_summary
from app.services.token_counter import estimate_messages_tokens
from app.tools.registry import get_all_tools

HISTORY_MESSAGE_LIMIT = 50
HISTORY_TOKEN_BUDGET = 6000


async def chat(
        db: AsyncSession,
        redis: Redis,
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
    # 3. 如果历史消息达到条件，先更新Summary
    await update_summary(
        db=db,
        conversation=conversation
    )
    # 4. 优先从Redis获取短期Memory
    history = await get_recent_messages(
        redis=redis,
        conversation_id=conversation.id
    )
    # 5. Redis未命中，从PostgreSQL恢复
    if not history:
        history = await get_messages(
            db=db,
            conversation_id=conversation.id,
            limit=HISTORY_MESSAGE_LIMIT
        )
        # PostgreSQL是长期Memory的Source of Truth。
        # 回溯成功后重新建立Redis Short-term Memory。
        await rebuild_memory(
            redis=redis,
            conversation_id=conversation.id,
            messages=history
        )
    # 6. 根据 Token Budget 筛选历史
    history = select_messages_by_token_budget(
        messages=history,
        token_budget=HISTORY_TOKEN_BUDGET,
        token_counter=estimate_messages_tokens
    )
    # 7. 创建当前用户消息
    current_user_message = HumanMessage(
        content=message
    )
    # 8. 构造 Agent Context
    messages = history.copy()
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
    messages.append(current_user_message)
    # 9. 创建LangChain Agent
    system_prompt = (
        '你是一个专业的 AI 助手。'
        '回答用户问题时要准确、简洁。'
        '如果需要查询实时天气，可以使用天气工具。'
        '如果用户的问题涉及内部知识、业务规则、'
        '产品文档或知识库内容，可以优先使用知识库搜索工具。'
        '用户不满意回答，再扩展搜索资料补充。'
    )
    agent = create_agent(
        model=create_llm(),
        tools=get_all_tools(db=db),
        system_prompt=system_prompt
    )
    # 10. 调用 Agent
    result = await agent.ainvoke({
        'messages': messages
    })
    result_messages = result['messages']
    # 11. 找到当前User Message在Agent返回结果中的位置
    current_user_index = next(
        index
        for index, msg in enumerate(result_messages)
        if msg is current_user_message
    )
    # 12. 只保存当前User Message后面的Agent消息
    new_messages = result_messages[current_user_index:]
    for msg in new_messages:
        await save_langchain_message(
            db=db,
            conversation_id=conversation.id,
            message=msg
        )
    # 13. Redis更新Short-term Memory
    await save_messages(
        redis=redis,
        conversation_id=conversation.id,
        messages=new_messages
    )
    # 14. 最后一条消息就是回答
    answer = result_messages[-1].content

    return conversation.id, answer
