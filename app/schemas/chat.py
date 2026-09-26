from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatAttachment(BaseModel):
    file_id: int


class ChatRequest(BaseModel):
    conversation_id: int | None = None
    message: str
    # 每次创建 ChatRequest 时生成一个新的 list, 避免可变默认值
    attachments: list[ChatAttachment] = Field(default_factory=list)


class ChatResumeRequest(BaseModel):
    conversation_id: int
    interrupt_id: str
    approved: bool


class ChatResponse(BaseModel):
    conversation_id: int
    answer: str
    status: Literal['completed', 'waiting_approval', 'need_more_info'] = 'completed'
    intent: Literal['knowledge', 'ticket', 'general', 'multimodal'] | None = None
    ticket_id: int | None = None
    interrupt_id: str | None = None
    interrupt_value: dict[str, Any] | None = None
