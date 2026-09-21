from typing import Any

from langchain_core.messages import SystemMessage, AnyMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.errors import NodeError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy, default_retry_on
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.checkpointer import checkpointer
from app.core.llm import create_llm
from app.tools.registry import get_all_tools

SYSTEM_PROMPT = (
    "你是一个专业的 AI 助手。"
    "回答用户问题时要准确、简洁。"
    "如果需要查询实时天气，可以使用天气工具。"
    "如果用户的问题涉及内部知识、业务规则、产品文档或知识库内容，可以使用知识库搜索工具。"

    "如果用户明确要求删除知识库文档，"
    "你必须调用 delete_knowledge_document 工具。"
    "不要自行向用户询问确认。"
    "不要仅通过自然语言告诉用户需要确认。"
    "delete_knowledge_document 工具内部会负责人工审批。"
)


class AgentState(MessagesState):
    """
    LangGraph Agent 的 State。

    messages:
        LangGraph Agent 当前的完整执行消息状态。
        ToolNode 和 tools_condition 都依赖它。
    context_messages:
        每一次请求由 Redis/PostgreSQL Memory 计算出来的
        当前 LLM 上下文，不负责 Graph 的执行状态。
    summary:
        当前最新的长期对话摘要
    """
    context_messages: list[AnyMessage]
    summary: str | None


# Agent Node的RetryPolicy
# initial_interval: 第一次失败以后，等多少秒再进行第一次重试
# backoff_factor: 指数退避（exponential backoff），每次重试之间的等待时间乘以多少
# max_interval: 最大的等待时间。min(计算出来的 backoff, max_interval)
# max_attempts: 总共最多执行次数，包含第一次
# jitter: 是否给重试等待时间增加一点随机性。Thundering Herd（惊群），大量请求在同一时间再次打向已经有问题的服务
# retry_on: default_retry_on，LangGraph 使用自己的默认“哪些异常适合 Retry”的判断
AGENT_RETRY_POLICY = RetryPolicy(
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=4.0,
    max_attempts=3,
    jitter=False,
    retry_on=default_retry_on
)


def handle_agent_error(
        state: AgentState,
        error: NodeError
) -> dict:
    """
    Agent Node 多次尝试仍然失败后的最终错误处理。
    """
    print(
        f'Agent Node 执行失败：'
        f'node={error.node}, '
        f'error={error.error}'
    )

    return {
        'messages': [
            AIMessage(
                content=(
                    '抱歉，模型服务当前暂时不可用，'
                    '请稍后再试。'
                )
            )
        ]
    }


def add_agent_branch(
        builder: Any,
        db: AsyncSession,
        tools: list[BaseTool] | None = None
) -> None:
    """
    向外层Workflow注册通用Agent分支。

    负责：
    - 普通对话
    - 天气查询
    - 删除知识库等需要LLM自主决定工具调用的场景
    """
    if tools is None:
        tools = get_all_tools(db=db)
    model = create_llm().bind_tools(tools)

    async def call_model(
            state: AgentState
    ) -> dict:
        current_turn_start = 0
        for index in range(len(state['messages']) - 1, -1, -1):
            if isinstance(state['messages'][index], HumanMessage):
                current_turn_start = index
                break
        current_turn_messages = state['messages'][current_turn_start:]
        prompt_messages = [
            SystemMessage(
                content=SYSTEM_PROMPT
            )
        ]
        if state.get('summary'):
            prompt_messages.append(
                SystemMessage(
                    content=(
                        "以下是当前会话较早历史的摘要。"
                        "它用于帮助你理解长期上下文。"
                        "如果摘要与最近的原始消息存在冲突，"
                        "优先相信最近的原始消息。\n\n"
                        f"会话摘要：\n{state['summary']}"
                    )
                )
            )
        prompt_messages.extend(state.get('context_messages', []))
        prompt_messages.extend(current_turn_messages)
        response = await model.ainvoke(prompt_messages)

        return {
            'messages': [response]
        }

    builder.add_node(
        'agent',
        call_model,
        retry_policy=AGENT_RETRY_POLICY,
        error_handler=handle_agent_error
    )
    builder.add_node(
        'tools',
        ToolNode(
            tools,
            handle_tool_errors=True
        )
    )
    builder.add_conditional_edges(
        'agent',
        tools_condition,
        {
            'tools': 'tools',
            '__end__': END
        }
    )
    builder.add_edge('tools', 'agent')


def create_agent_graph(
        db: AsyncSession
):
    """
    独立创建通用 Agent Graph。
    保留这个函数，方便后续单独测试 Agent。
    """
    builder = StateGraph(AgentState)
    add_agent_branch(builder, db)
    builder.add_edge(START, 'agent')

    return builder.compile(
        checkpointer=checkpointer
    )
