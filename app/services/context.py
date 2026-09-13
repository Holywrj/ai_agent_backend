from collections.abc import Callable

from langchain_core.messages import BaseMessage

# 接收 list[BaseMessage]，返回 int 的函数
MessageTokenCounter = Callable[[list[BaseMessage]], int]


def select_messages_by_token_budget(
        messages: list[BaseMessage],
        token_budget: int,
        token_counter: MessageTokenCounter
) -> list[BaseMessage]:
    """
    根据Token Budget选择历史消息
    从最新的完整对话轮次开始向前选择，直到继续加入下一轮会超过Token Budget。
    :param messages: 按时间顺序排列的历史消息
    :param token_budget: 历史消息允许使用的最大Token数
    :param token_counter: LangChain Message Token计算函数
    :return: 保持时间顺序的上下文消息
    """
    if token_budget <= 0:
        return []
    if not messages:
        return []
    # 按user消息划分完整的对话轮次
    turns: list[list[BaseMessage]] = []
    current_turn: list[BaseMessage] = []
    for message in messages:
        if message.type == 'human':
            if current_turn:
                turns.append(current_turn)
            current_turn = [message]
        else:
            current_turn.append(message)
    if current_turn:
        turns.append(current_turn)
    selected_turns: list[list[BaseMessage]] = []
    current_tokens = 0
    # 从最新一轮开始向前选择
    for turn in turns:
        turn_tokens = token_counter(turn)
        if not selected_turns:
            # 最新一轮即使超过预算，也保留
            # 否则可能导致当前对话完全没有上下文
            selected_turns.append(turn)
            current_tokens += turn_tokens
            continue
        if current_tokens + turn_tokens > token_budget:
            break
        selected_turns.append(turn)
        current_tokens += turn_tokens
    # 恢复时间顺序
    selected_turns.reverse()

    return [
        message
        for turn in selected_turns
        for message in turn
    ]
