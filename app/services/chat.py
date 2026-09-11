from langchain.agents import create_agent

from app.core.checkpointer import checkpointer
from app.core.llm import create_llm
from app.tools.registry import get_all_tools


async def chat(
        message: str,
        user_id: int,
        conversation_id: str
) -> str:
    llm = create_llm()
    # LangChain Agent 短期记忆，本质上是线程级（thread-level）的 Agent 状态持久化；
    # Agent 默认用 messages 保存对话历史，要让这些状态跨多次调用保留下来，需要给 create_agent() 配置 checkpointer，
    # 并在每次调用时提供同一个 thread_id
    # short-term memory 模式
    # Checkpointer 负责保存和恢复 Agent 的状态；它不会替 LLM 理解用户，也不会决定不同用户是否隔离。真正决定隔离的是你传入的 thread_id
    agent = create_agent(
        model=llm,
        tools=get_all_tools(),
        system_prompt=(
            '你是一个专业的 AI 助手。'
            '回答用户问题时要准确、简洁。'
            '如果需要查询实时天气，可以使用天气工具。'
        ),
        checkpointer=checkpointer
    )
    thread_id = f'{user_id}:{conversation_id}'

    result = await agent.ainvoke(
        {
            'messages':[{
                'role': 'user',
                'content': message
            }]
        },
        config={
            'configurable': {
                'thread_id': thread_id
            }
        }
    )

    return result['messages'][-1].content
