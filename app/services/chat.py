from langchain_core.messages import HumanMessage
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.checkpointer import checkpointer
from app.exceptions.base import BusinessException
from app.graph.agent import create_agent_graph
from app.models.conversation import Conversation
from app.services.context import select_messages_by_token_budget
from app.services.memory import create_conversation, get_messages, save_langchain_message
from app.services.redis_memory import get_recent_messages, rebuild_memory, save_messages
from app.services.summary import update_summary
from app.services.token_counter import estimate_messages_tokens

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
    # 8. 创建LangGraph
    graph = create_agent_graph(db=db)
    # 9. conversation_id -> LangGraph thread_id
    # 当前业务层已经保证Conversation属于当前user，
    # 因此conversation_id可以作为Graph状态轨迹的唯一业务标识。
    thread_id = f'conversation:{conversation.id}'
    config = {
        'configurable': {
            'thread_id': thread_id
        }
    }
    # 10. 查询这个thread是否已经存在checkpoint
    checkpoint = await graph.aget_state(
        config
    )
    checkpoint_messages = checkpoint.values.get('messages', [])
    has_checkpoint = bool(checkpoint_messages)
    if has_checkpoint:
        # 已经存在checkpoint
        # graph会恢复以前的agent state，因此这里只传本次新增的HumanMessage
        initial_message_count = len(checkpoint_messages)
        graph_input = {
            'messages': [current_user_message],
            'context_messages': history,
            'summary': conversation.summary,
        }
    else:
        # 第一次使用这个thread
        # checkpointer没有以前的state，所以用业务Memory给graph初始话工作状态
        initial_message_count = len(history)
        graph_input = {
            'messages': [*history, current_user_message],
            'context_messages': history,
            'summary': conversation.summary,
        }
    # 11. 调用 LangGraph
    result = await graph.ainvoke(
        graph_input,
        config
    )
    result_messages = result['messages']
    # 12. 只保存本次调用新产生的消息
    new_messages = result_messages[initial_message_count:]
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
