from langgraph.checkpoint.memory import InMemorySaver

# 模块级单例对象
# Checkpointer 是 Agent 状态持久化机制， InMemorySaver 是它的实现
# LangChain Agent 短期记忆，本质上是线程级（thread-level）的 Agent 状态持久化；
# Agent 默认用 messages 保存对话历史，要让这些状态跨多次调用保留下来，需要给 create_agent() 配置 checkpointer，
# 并在每次调用时提供同一个 thread_id
# short-term memory 模式
# Checkpointer 负责保存和恢复 Agent 的状态；它不会替 LLM 理解用户，也不会决定不同用户是否隔离。真正决定隔离的是你传入的 thread_id
checkpointer = InMemorySaver()
