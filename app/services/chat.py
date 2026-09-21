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
from app.graph.workflow import create_workflow_graph
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
    status: Literal['completed', 'waiting_approval', 'need_more_info']
    intent: Literal['knowledge', 'ticket', 'general'] | None = None
    ticket_id: int | None = None
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


async def _prepare_chat_context(
        db: AsyncSession,
        redis: Redis,
        message: str,
        user_id: int,
        conversation_id: int | None
):
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
    await update_summary(
        db=db,
        conversation=conversation
    )
    history = await get_recent_messages(
        redis=redis,
        conversation_id=conversation.id
    )
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
    history = select_messages_by_token_budget(
        messages=history,
        token_budget=HISTORY_TOKEN_BUDGET,
        token_counter=estimate_messages_tokens
    )

    return (
        conversation,
        history,
        HumanMessage(content=message)
    )


async def _persist_messages(
        db: AsyncSession,
        redis: Redis,
        conversation_id: int,
        messages: list[Any]
) -> None:
    for msg in messages:
        await save_langchain_message(
            db=db,
            conversation_id=conversation_id,
            message=msg,
        )
    await save_messages(
        redis=redis,
        conversation_id=conversation_id,
        messages=messages
    )


def _normal_status(value: Any) -> Literal['completed', 'need_more_info']:
    if value == 'need_more_info':
        return 'need_more_info'
    return 'completed'


def _make_sse_event(
        event: str,
        data: dict[str, Any]
) -> str:
    return (
        f'event: {event}\n'
        f'data: '
        f'{json.dumps(data, ensure_ascii=False)}\n\n'
    )


# todo: 普通非流式 Chat
async def chat(
        db: AsyncSession,
        redis: Redis,
        message: str,
        user_id: int,
        conversation_id: int | None
) -> ChatResult:
    conversation, history, current_user_message = await _prepare_chat_context(
        db=db,
        redis=redis,
        message=message,
        user_id=user_id,
        conversation_id=conversation_id
    )
    graph = create_workflow_graph(db=db)
    thread_id = f'conversation:{conversation.id}'
    config = {
        'configurable': {
            'thread_id': thread_id
        }
    }
    checkpoint = await graph.aget_state(config)
    if checkpoint.interrupts:
        raise BusinessException(
            message='Conversation has a pending approval',
            code='APPROVAL_PENDING'
        )
    checkpoint_messages = checkpoint.values.get('messages', [])
    has_checkpoint = bool(checkpoint_messages)
    if has_checkpoint:
        initial_message_count = len(checkpoint_messages)
        graph_input = {
            'messages': [current_user_message],
            'context_messages': history,
            'summary': conversation.summary,
            'user_id': user_id,
            'conversation_id': conversation.id
        }
    else:
        initial_message_count = len(history)
        graph_input = {
            'messages': [*history, current_user_message],
            'context_messages': history,
            'summary': conversation.summary,
            'user_id': user_id,
            'conversation_id': conversation.id
        }
    result = await graph.ainvoke(
        graph_input,
        config
    )
    interrupts = result.get('__interrupt__', ())
    if interrupts:
        interrupt_info = interrupts[0]
        interrupted_state = await graph.aget_state(config)
        interrupted_messages = interrupted_state.values.get('messages', [])
        new_messages = interrupted_messages[initial_message_count:]
        await _persist_messages(
            db=db,
            redis=redis,
            conversation_id=conversation.id,
            messages=new_messages
        )
        return ChatResult(
            conversation_id=conversation.id,
            answer='等待人工确认。',
            status='waiting_approval',
            intent=result.get('intent'),
            ticket_id=result.get('ticket_id'),
            interrupt_id=interrupt_info.id,
            interrupt_value=interrupt_info.value
        )
    result_messages = result.get('messages', [])
    new_messages = result_messages[initial_message_count:]
    await _persist_messages(
        db=db,
        redis=redis,
        conversation_id=conversation.id,
        messages=new_messages
    )
    return ChatResult(
        conversation_id=conversation.id,
        answer=result_messages[-1].content,
        status=_normal_status(result.get('workflow_status')),
        intent=result.get('intent'),
        ticket_id=result.get('ticket_id')
    )


# todo: Resume
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
    graph = create_workflow_graph(db=db)
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
            intent=result.get('intent'),
            ticket_id=result.get('ticket_id'),
            interrupt_id=interrupt_info.id,
            interrupt_value=interrupt_info.value
        )

    result_messages = result.get('messages', [])
    new_messages = result_messages[before_message_count:]
    await _persist_messages(
        db=db,
        redis=redis,
        conversation_id=conversation.id,
        messages=new_messages
    )

    return ChatResult(
        conversation_id=conversation.id,
        answer=result_messages[-1].content,
        status=_normal_status(result.get('workflow_status')),
        intent=result.get('intent'),
        ticket_id=result.get('ticket_id')
    )


# todo: Streaming
async def stream_chat(
        db: AsyncSession,
        redis: Redis,
        message: str,
        user_id: int,
        conversation_id: int | None
) -> AsyncIterator[str]:
    # 1. 数据准备
    conversation, history, current_user_message = await _prepare_chat_context(
        db=db,
        redis=redis,
        message=message,
        user_id=user_id,
        conversation_id=conversation_id
    )
    # 2. 创建Graph
    graph = create_workflow_graph(db=db)
    thread_id = f'conversation:{conversation.id}'
    config = {
        'configurable': {
            'thread_id': thread_id
        }
    }
    # 3. 检查当前Graph State
    checkpoint = await graph.aget_state(config)
    if checkpoint.interrupts:
        yield _make_sse_event(
            'error',
            {
                'code': 'APPROVAL_PENDING',
                'message': 'Conversation has a pending approval'
            }
        )
        return
    checkpoint_messages = checkpoint.values.get('messages', [])
    has_checkpoint = bool(checkpoint_messages)
    # 4. 准备本次Graph输入
    if has_checkpoint:
        initial_message_count = len(checkpoint_messages)
        graph_input = {
            'messages': [current_user_message],
            'context_messages': history,
            'summary': conversation.summary,
            'user_id': user_id,
            'conversation_id': conversation.id
        }
    else:
        initial_message_count = len(history)
        graph_input = {
            'messages': [*history, current_user_message],
            'context_messages': history,
            'summary': conversation.summary,
            'user_id': user_id,
            'conversation_id': conversation.id
        }
    # 5. Graph Streaming
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
            # Workflow中Router / Ticket Extract也可能产生内部LLM stream。
            # 这些不是给用户展示的token。
            # 只把真正面向用户的Agent / Knowledge Answer的token推给前端
            if metadata.get('langgraph_node') not in {'agent', 'knowledge_answer'}:
                continue
            if isinstance(message_chunk, AIMessageChunk):
                if isinstance(message_chunk.content, str) and message_chunk.content:
                    yield _make_sse_event(
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
                yield _make_sse_event(
                    'node',
                    {
                        'node': node_name
                    }
                )
            # Interrupt
            interrupts = updates.get('__interrupt__')
            if interrupts:
                interrupt_info = interrupts[0]
                yield _make_sse_event(
                    'interrupt',
                    {
                        'conversation_id': conversation.id,
                        'interrupt_id': interrupt_info.id,
                        'interrupt_value': interrupt_info.value
                    }
                )
    # 6. Streaming结束后，从Checkpoint获取最终State
    final_state = await graph.aget_state(config)
    result_messages = final_state.values.get('messages', [])
    # 7. Graph当前是否停在Interrupt
    if final_state.interrupts:
        new_messages = result_messages[initial_message_count:]
        await _persist_messages(
            db=db,
            redis=redis,
            conversation_id=conversation.id,
            messages=new_messages
        )
        yield _make_sse_event(
            'done',
            {
                'conversation_id': conversation.id,
                'status': 'waiting_approval',
                'intent': final_state.values.get('intent'),
                'ticket_id': final_state.values.get('ticket_id')
            }
        )
        return
    # 8. 正常结束，保存本次消息
    new_messages = result_messages[initial_message_count:]
    await _persist_messages(
        db=db,
        redis=redis,
        conversation_id=conversation.id,
        messages=new_messages
    )
    yield _make_sse_event(
        'done',
        {
            'conversation_id': conversation.id,
            'status': _normal_status(final_state.values.get('workflow_status')),
            'intent': final_state.values.get('intent'),
            'ticket_id': final_state.values.get('ticket_id')
        }
    )
