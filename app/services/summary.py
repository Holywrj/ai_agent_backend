from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import BaseMessage

from app.core.llm import create_llm
from app.models.conversation import Conversation
from app.models.message import Message
from app.services.memory import to_langchain_message

SUMMARY_TRIGGER_MESSAGES = 30
SUMMARY_KEEP_RECENT_TURNS = 3


async def get_summary_messages(
        db: AsyncSession,
        conversation: Conversation
) -> list[Message]:
    """
    获取尚未进入Summary的消息。
    """
    query = (
        select(Message)
        .where(
            Message.conversation_id == conversation.id
        )
        .order_by(
            Message.created_at,
            Message.id
        )
    )
    if conversation.summary_message_id is not None:
        query = query.where(
            Message.id > conversation.summary_message_id
        )
    result = await db.execute(query)

    return list(result.scalars().all())


def split_messages_into_turns(
        messages: list[Message]
) -> list[list[Message]]:
    """
    按用户消息划分完整对话轮次
    一轮可能包含：user, assistant(tool_call), tool, assistant(final)
    """
    turns: list[list[Message]] = []
    current_turn: list[Message] = []
    for message in messages:
        if message.role == 'user':
            if current_turn:
                turns.append(current_turn)
            current_turn = [message]
        else:
            current_turn.append(message)
    if current_turn:
        turns.append(current_turn)

    return turns


def select_messages_for_summary(
        messages: list[Message]
) -> list[Message]:
    """
    选择应该进入Summry的完整对话轮次。
    保留最近SUMMARY_KEEP_RECENT_TURNS轮，
    避免Summary和当前上下文发生重叠。
    """
    turns = split_messages_into_turns(messages)
    if len(turns) < SUMMARY_KEEP_RECENT_TURNS:
        return []
    turns_to_summarize = turns[:-SUMMARY_KEEP_RECENT_TURNS]

    return [
        message
        for turn in turns_to_summarize
        for message in turn
    ]


async def update_summary(
        db: AsyncSession,
        conversation: Conversation,
) -> None:
    """
    增量更新Conversation Summary。
    只处理尚未总结的旧对话，最近几轮继续保留为原始消息。
    """
    messages = await get_summary_messages(
        db=db,
        conversation=conversation
    )
    if len(messages) <= SUMMARY_TRIGGER_MESSAGES:
        return
    # 最近消息继续作为精确上下文，不进入Summary
    messages_to_summarize = select_messages_for_summary(messages)
    if not messages_to_summarize:
        return
    # PostgreSQL Message -> LangChain Message
    langchain_messages: list[BaseMessage] = [
        to_langchain_message(message)
        for message in messages_to_summarize
    ]
    new_messages_text = '\n'.join(
        f'{message.type}: {message.content}'
        for message in langchain_messages
    )
    existing_summary = conversation.summary or '暂无历史摘要'
    prompt = f"""
        你负责维护一个 AI Agent 的长期对话摘要。
    
        请根据已有摘要和新增的历史消息，生成一个新的、准确且简洁的对话摘要。
    
        要求：
    
        1. 保留用户明确表达的重要信息、需求、偏好和约束。
        2. 保留已经确定的重要事实、结论和决策。
        3. 保留当前正在进行的任务及其进展。
        4. 不要记录无关的闲聊。
        5. 不要编造消息中不存在的信息。
        6. 新摘要应该能够让另一个 AI Agent 在不知道完整历史消息的情况下，
           理解这个会话的重要背景。
        7. 输出纯文本摘要，不要添加标题、Markdown 或解释。
    
        已有摘要：
        {existing_summary}
    
        新增历史消息：
        {new_messages_text}
    """
    llm = create_llm()
    response = await llm.ainvoke(prompt)
    conversation.summary = response.content
    conversation.summary_message_id = messages_to_summarize[-1].id
    await db.commit()
    await db.refresh(conversation)
