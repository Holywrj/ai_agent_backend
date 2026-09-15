import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from redis.asyncio import Redis

REDIS_MEMORY_TTL = 1800
REDIS_MEMORY_MAX_MESSAGES = 50

def build_memory_key(conversation_id: int) -> str:
    return f'agent:memory:conversation:{conversation_id}'

def serialize_message(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        data = {
            'type': 'human',
            'content': message.content
        }
    elif isinstance(message, AIMessage):
        data = {
            'type': 'ai',
            'content': message.content,
            'tool_calls': message.tool_calls or []
        }
    elif isinstance(message, ToolMessage):
        data = {
            'type': 'tool',
            'content': message.content,
            'tool_call_id': message.tool_call_id
        }
    else:
        raise ValueError(
            f'Unsupported message type: {type(message)}'
        )

    return json.dumps(
        data,
        ensure_ascii=False,
        default=str
    )


def deserialize_message(data: str) -> BaseMessage:
    message = json.loads(data)
    message_type = message['type']
    if message_type == 'human':
        return HumanMessage(
            content=message['content']
        )
    if message_type == 'ai':
        return AIMessage(
            content=message['content'],
            tool_calls=message.get('tool_calls') or []
        )
    if message_type == 'tool':
        return ToolMessage(
            content=message['content'],
            tool_call_id=message['tool_call_id']
        )
    raise ValueError(
        f'Unsupported Redis message type: {message_type}'
    )


def select_complete_recent_messages(
        messages: list[BaseMessage]
) -> list[BaseMessage]:
    """
        选择最近的完整对话轮次。

        REDIS_MEMORY_MAX_MESSAGES 是数量上限，但不会强行
        从某个 AIMessage / ToolMessage 中间截断。

        正常情况下：
            从最近 50 条的起点向后寻找下一个 HumanMessage，
            从这个 HumanMessage 开始保留。

        特殊情况下：
            如果最近 50 条全部属于同一个超长 Turn，
            则向前寻找这个 Turn 的 HumanMessage，
            保留完整的最后一轮，即使最终超过 50 条。
    """
    if not messages:
        return []
    if len(messages) <= REDIS_MEMORY_MAX_MESSAGES:
        return messages

    start_index = len(messages) - REDIS_MEMORY_MAX_MESSAGES
    if messages[start_index].type == 'human':
        return messages[start_index:]
    # 当起点在某个turn中间，优先向后找下一个HumanMessage。
    for index in range(start_index + 1, len(messages)):
        if messages[index].type == 'human':
            return messages[index:]
    # 如果后面已经没有HumanMessage，说明最后一个turn超过了数量上限，
    # 此时向前找到这个turn的HumanMessage。
    for index in range(start_index - 1, -1, -1):
        if messages[index].type == 'human':
            return messages[index:]
    return messages


async def get_recent_messages(
        redis: Redis,
        conversation_id: int
) -> list[BaseMessage]:
    key = build_memory_key(conversation_id)
    raw_messages = await redis.lrange(key, 0, -1)
    messages = [
        deserialize_message(data)
        for data in raw_messages
    ]
    return select_complete_recent_messages(messages)


async def save_messages(
        redis: Redis,
        conversation_id: int,
        messages: list[BaseMessage]
) -> None:
    if not messages:
        return
    key = build_memory_key(conversation_id)
    raw_messages = await redis.lrange(key, 0, -1)
    current_messages = [
        deserialize_message(data)
        for data in raw_messages
    ]
    all_messages = [
        *current_messages,
        *messages
    ]
    complete_messages = select_complete_recent_messages(all_messages)
    serialized_messages = [
        serialize_message(message)
        for message in complete_messages
    ]

    async with redis.pipeline(transaction=True) as pipe:
        pipe.delete(key)
        if serialized_messages:
            pipe.rpush(key, *serialized_messages)
            pipe.expire(key, REDIS_MEMORY_TTL)
        await pipe.execute()


async def rebuild_memory(
        redis: Redis,
        conversation_id: int,
        messages: list[BaseMessage]
) -> None:
    key = build_memory_key(conversation_id)
    complete_messages = select_complete_recent_messages(messages)
    serialized_messages = [
        serialize_message(message)
        for message in complete_messages
    ]

    async with redis.pipeline(transaction=True) as pipe:
        pipe.delete(key)
        if serialized_messages:
            pipe.rpush(key, *serialized_messages)
            pipe.expire(key, REDIS_MEMORY_TTL)
        await pipe.execute()


async def delete_memory(
        redis: Redis,
        conversation_id: int
) -> None:
    key = build_memory_key(conversation_id)
    await redis.delete(key)
