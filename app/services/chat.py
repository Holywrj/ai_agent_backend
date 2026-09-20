from dataclasses import dataclass
from typing import Any, Literal
import json
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.types import Command
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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


@dataclass(slots=True)
class ChatResult:
    conversation_id: int
    answer: str
    status: Literal['completed', 'waiting_approval']
    interrupt_id: str | None = None
    interrupt_value: dict[str, Any] | None = None


async def _get_conversation(
        db: AsyncSession,
        conversation_id: int,
        user_id: int
) -> Conversation:
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
    return conversation


async def chat(
        db: AsyncSession,
        redis: Redis,
        message: str,
        user_id: int,
        conversation_id: int | None
) -> ChatResult:
    # 1. 获取/创建 Conversation
    if conversation_id is None:
        conversation = await create_conversation(
            db=db,
            user_id=user_id
        )
    else:
        conversation = await _get_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id
        )
    # 2. 如果历史消息达到条件，先更新Summary
    await update_summary(
        db=db,
        conversation=conversation
    )
    # 3. 优先从Redis获取短期Memory
    history = await get_recent_messages(
        redis=redis,
        conversation_id=conversation.id
    )
    # 4. Redis未命中，从PostgreSQL恢复
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
    # 5. 根据 Token Budget 筛选历史
    history = select_messages_by_token_budget(
        messages=history,
        token_budget=HISTORY_TOKEN_BUDGET,
        token_counter=estimate_messages_tokens
    )
    # 6. 创建当前用户消息
    current_user_message = HumanMessage(
        content=message
    )
    # 7. 创建LangGraph
    graph = create_agent_graph(db=db)
    thread_id = f'conversation:{conversation.id}'
    config = {
        'configurable': {
            'thread_id': thread_id
        }
    }
    # 8. 检查当前Graph State
    checkpoint = await graph.aget_state(
        config
    )
    # 如果之前已经暂停等待审批，不允许用户绕过审批继续提交新的消息。
    if checkpoint.interrupts:
        raise BusinessException(
            message='Conversation has a pending approval',
            code='APPROVAL_PENDING'
        )
    checkpoint_messages = checkpoint.values.get('messages', [])
    has_checkpoint = bool(checkpoint_messages)
    # 9. 确定本次Graph执行开始之前已有多少消息
    if has_checkpoint:
        initial_message_count = len(checkpoint_messages)
        graph_input = {
            'messages': [current_user_message],
            'context_messages': history,
            'summary': conversation.summary,
        }
    else:
        initial_message_count = len(history)
        graph_input = {
            'messages': [*history, current_user_message],
            'context_messages': history,
            'summary': conversation.summary,
        }
    # 10. 执行Graph
    result = await graph.ainvoke(
        graph_input,
        config
    )
    # 11. 检查是否被interrupt
    interrupts = result.get('__interrupt__', ())
    if interrupts:
        interrupt_info = interrupts[0]
        # 通过Checkpoint读取中断时真正保存下来的State
        interrupted_state = await graph.aget_state(config)
        interrupted_messages = interrupted_state.values.get('messages', [])
        # 保存本次已经产生的业务信息 Human + AI（tool_call）
        new_messages = interrupted_messages[initial_message_count:]
        for msg in new_messages:
            await save_langchain_message(
                db=db,
                conversation_id=conversation.id,
                message=msg
            )
        await save_messages(
            redis=redis,
            conversation_id=conversation.id,
            messages=new_messages
        )

        return ChatResult(
            conversation_id=conversation.id,
            answer='等待人工确认。',
            status='waiting_approval',
            interrupt_id=interrupt_info.id,
            interrupt_value=interrupt_info.value
        )

    # 12. Graph正常结束
    result_messages = result['messages']
    new_messages = result_messages[initial_message_count:]
    for msg in new_messages:
        await save_langchain_message(
            db=db,
            conversation_id=conversation.id,
            message=msg
        )
    await save_messages(
        redis=redis,
        conversation_id=conversation.id,
        messages=new_messages
    )

    return ChatResult(
        conversation_id=conversation.id,
        answer=result_messages[-1].content,
        status='completed'
    )


async def resume_chat(
        db: AsyncSession,
        redis: Redis,
        conversation_id: int,
        user_id: int,
        interrupt_id: str,
        approved: bool
) -> ChatResult:
    # 1. 校验Conversation所属用户
    conversation = await _get_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=user_id
    )
    # 2. 创建Graph
    graph = create_agent_graph(db=db)
    thread_id = f'conversation:{conversation.id}'
    config = {
        'configurable': {
            'thread_id': thread_id
        }
    }
    # 3. 获取当前Checkpoint
    checkpoint = await graph.aget_state(config)
    if not checkpoint.interrupts:
        raise BusinessException(
            message='No pending approval',
            code='NO_PENDING_APPROVAL'
        )
    pending_interrupt = next(
        (
            item
            for item in checkpoint.interrupts
            if item.id == interrupt_id
        ),
        None
    )
    if pending_interrupt is None:
        raise BusinessException(
            message='Interrupt not found',
            code='INTERRUPT_NOT_FOUND'
        )
    # 4. Resume
    before_message_count = len(checkpoint.values.get('messages', []))
    result = await graph.ainvoke(
        Command[Any](
            resume=approved
        ),
        config
    )
    # 5. 正常结束
    interrupts = result.get('__interrupt__', ())
    if interrupts:
        interrupt_info = interrupts[0]

        return ChatResult(
            conversation_id=conversation.id,
            answer='仍然等待人工确认',
            status='waiting_approval',
            interrupt_id=interrupt_info.id,
            interrupt_value=interrupt_info.value
        )

    result_messages = result['messages']
    new_messages = result_messages[before_message_count:]
    for msg in new_messages:
        await save_langchain_message(
            db=db,
            conversation_id=conversation.id,
            message=msg
        )
    await save_messages(
        redis=redis,
        conversation_id=conversation.id,
        messages=new_messages
    )

    return ChatResult(
        conversation_id=conversation.id,
        answer=result_messages[-1].content,
        status='completed'
    )


async def stream_chat(
        db: AsyncSession,
        redis: Redis,
        message: str,
        user_id: int,
        conversation_id: int | None
) -> AsyncIterator[str]:
    # 1 获取/创建Conversation
    if conversation_id is None:
        conversation = await create_conversation(
            db=db,
            user_id=user_id
        )
    else:
        conversation = await _get_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id
        )
    # 2. 更新Summary
    await update_summary(
        db=db,
        conversation=conversation
    )
    # 3. Redis获取短期Memory
    history = await get_recent_messages(
        redis=redis,
        conversation_id=conversation.id
    )
    # 4. Redis MISS -> PostgreSQL
    if not history:
        history = await get_messages(
            db=db,
            conversation_id=conversation.id,
            limit=HISTORY_MESSAGE_LIMIT
        )
        await rebuild_memory(
            redis=redis,
            conversation_id=conversation.id,
            messages=history
        )
    # 5. Token Budget
    history = select_messages_by_token_budget(
        messages=history,
        token_budget=HISTORY_TOKEN_BUDGET,
        token_counter=estimate_messages_tokens
    )
    # 6. 当前User Message
    current_user_message = HumanMessage(
        content=message
    )
    # 7. 创建Graph
    graph = create_agent_graph(db=db)
    thread_id = f'conversation:{conversation.id}'
    config = {
        'configurable': {
            'thread_id': thread_id
        }
    }
    # 8. 检查当前Graph State
    checkpoint = await graph.aget_state(config)
    if checkpoint.interrupts:
        yield (
            'event: error\n'
            'data: '
            + json.dumps(
                {
                    'code': 'APPROVAL_PENDING',
                    'message': 'Conversation has a pending approval'
                },
                ensure_ascii=False
            )
            + '\n\n'
        )
        return
    checkpoint_messages = checkpoint.values.get('messages', [])
    has_checkpoint = bool(checkpoint_messages)
    # 9. 准备本次Graph输入
    if has_checkpoint:
        initial_message_count = len(checkpoint_messages)
        graph_input = {
            'messages': [current_user_message],
            'context_messages': history,
            'summary': conversation.summary
        }
    else:
        initial_message_count = len(history)
        graph_input = {
            'messages': [*history, current_user_message],
            'context_messages': history,
            'summary': conversation.summary
        }

    # 10. SSE 辅助函数
    def make_event(event: str, data: dict) -> str:
        return (
            f'event: {event}\n'
            f'data: {json.dumps(data, ensure_ascii=False)}\n\n'
        )

    # 11. Graph Streaming
    # astream()支持多个stream_mode同时使用
    # messages：LLM token流
    # updates：节点更新流
    # version：LangGraph Streaming 返回的数据结构采用哪一代 Stream Protocol
    #   v2 会把不同的流结果统一成带类型的信息结构
    async for part in graph.astream(
        graph_input,
        config,
        stream_mode=['messages', 'updates'],
        version='v2'
    ):
        # todo: messages: LLM Token Streaming
        if part['type'] == 'messages':
            message_chunk, metadata = part['data']
            if isinstance(message_chunk, AIMessageChunk):
                if isinstance(message_chunk.content, str) and message_chunk.content:
                    yield make_event(
                        'token',
                        {
                            'content': message_chunk.content
                        }
                    )
        # todo: updates: Graph / Node / Interrupt
        elif part['type'] == 'updates':
            updates = part['data']
            # 普通Node Update
            for node_name in updates:
                if node_name == '__interrupt__':
                    continue
                yield make_event(
                    'node',
                    {
                        'node': node_name
                    }
                )
            # Interrupt
            interrupts = updates.get('__interrupt__')
            if interrupts:
                interrupt_info = interrupts[0]
                yield make_event(
                    'interrupt',
                    {
                        'conversation_id': conversation.id,
                        'interrupt_id': interrupt_info.id,
                        'interrupt_value': interrupt_info.value
                    }
                )
    # 12. Streaming结束后，从Checkpoint获取最终State
    final_state = await graph.aget_state(config)
    result_messages = final_state.values.get('messages', [])
    # 13. Graph当前是否停在Interrupt
    if final_state.interrupts:
        new_messages = result_messages[initial_message_count:]
        for msg in new_messages:
            await save_langchain_message(
                db=db,
                conversation_id=conversation.id,
                message=msg
            )
        await save_messages(
            redis=redis,
            conversation_id=conversation.id,
            messages=new_messages
        )
        yield make_event(
            'done',
            {
                'conversation_id': conversation.id,
                'status': 'waiting_approval'
            }
        )
        return
    # 14. 正常结束，保存本次消息
    new_messages = result_messages[initial_message_count:]
    for msg in new_messages:
        await save_langchain_message(
            db=db,
            conversation_id=conversation.id,
            message=msg
        )
    await save_messages(
        redis=redis,
        conversation_id=conversation.id,
        messages=new_messages
    )
    yield make_event(
        'done',
        {
            'conversation_id': conversation.id,
            'status': 'completed'
        }
    )
