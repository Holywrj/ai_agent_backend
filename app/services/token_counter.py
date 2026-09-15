from langchain_core.messages import BaseMessage


def estimate_text_tokens(text: str) -> int:
    """
    粗略估算 DeepSeek token数。
    DeepSeek官方给出的经验值：
    - 英文字符 = 0.3 token
    - 中文字符 = 0.6 token
    这里只用于Context Window控制，不用于计费。
    实际token用量以DeepSeek API返回的 usage 为准。
    """
    tokens = 0.0
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            tokens += 0.6
        elif char.isascii():
            tokens += 0.3
        else:
            tokens += 0.6
    return max(1, int(tokens + 0.5))


def estimate_message_tokens(message: BaseMessage) -> int:
    """
    估算单条LangChain Message的token数
    """
    tokens = estimate_text_tokens(str(message.content))
    # Tool Calling的参数本身也是进入上下文的。
    if getattr(message, 'tool_calls', None):
        for tool_call in message.tool_calls:
            tokens += estimate_text_tokens(str(tool_call))
    # ToolMessage的tool_call_id也会进入消息结构。
    tool_call_id = getattr(message, 'tool_call_id', None)
    if tool_call_id:
        tokens += estimate_text_tokens(tool_call_id)
    return tokens


def estimate_messages_tokens(messages: list[BaseMessage]) -> int:
    """
    估算一组LangChain Message的token数。
    """
    return sum(
        estimate_message_tokens(message)
        for message in messages
    )
