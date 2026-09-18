from langchain_core.messages import SystemMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.checkpointer import checkpointer
from app.core.llm import create_llm
from app.tools.registry import get_all_tools

SYSTEM_PROMPT = (
    "你是一个专业的 AI 助手。"
    "回答用户问题时要准确、简洁。"
    "如果需要查询实时天气，可以使用天气工具。"
    "如果用户的问题涉及内部知识、业务规则、"
    "产品文档或知识库内容，可以优先使用知识库搜索工具。"
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


def create_agent_graph(
        db: AsyncSession
):
    # 1. 获取当前项目已有的Tools
    tools = get_all_tools(db=db)
    # 2. 让LLM具备Tool Calling能力
    model = create_llm().bind_tools(tools)

    # 3. Agent Node
    async def call_model(
            state: AgentState
    ) -> dict:
        # 找到本轮对话的起点。
        # Graph State中可能保存着以前很多轮消息，但本次调用真正新增的是最后一个HumanMessage
        # 以及后面的Tool/AI消息
        current_turn_start = 0
        for index in range(len(state['messages']) - 1, -1, -1):
            if isinstance(state['messages'][index], HumanMessage):
                current_turn_start = index
                break
        current_turn_message = state['messages'][current_turn_start:]
        prompt_messages = [
            SystemMessage(
                content=SYSTEM_PROMPT
            )
        ]
        # summary不放进message state，每次从当前state单独构造，避免摘要更新后使用旧摘要
        if state['summary']:
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
        # Redis / PostgreSQL 提供的当前上下文
        prompt_messages.extend(state['context_messages'])
        # 当前这一轮的Human / Tool / AI消息
        prompt_messages.extend(current_turn_message)

        response = await model.ainvoke(
            prompt_messages
        )
        return {
            'messages': [response]
        }

    # 4. 创建Graph Builder
    builder = StateGraph(AgentState)
    # 5. 注册Agent Node
    builder.add_node('agent', call_model)
    # 6. 注册ToolNode
    builder.add_node('tools', ToolNode(tools))
    # 7. Graph开始 -> Agent
    builder.add_edge(START, 'agent')
    # 8. Agent -> 根据Tool Calling结果决定下一步
    builder.add_conditional_edges(
        'agent',
        tools_condition,
        {
            'tools': 'tools',
            '__end__': END,
        }
    )
    # 9. Tool执行完 -> 回到Agent
    builder.add_edge('tools', 'agent')
    # 10. 编译Graph
    return builder.compile(
        checkpointer=checkpointer
    )
