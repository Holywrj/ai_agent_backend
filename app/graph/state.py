from typing import Any

from langgraph.graph import MessagesState

class BaseWorkflowState(MessagesState):
    """
    所有业务Workflow共享的基础State

    messages: LangGraph当前执行中的消息状态
    user_id: 当前用户ID
    conversation_id: 当前会话ID
    attachments: 当前请求关联的文件资源引用, 每个附近包含file_id、storage_key、mime_type
    """
    user_id: int
    conversation_id: int
    attachments: list[dict[str, Any]]
