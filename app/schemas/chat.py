from typing import Any, Literal

from pydantic import BaseModel


class ChatRequest(BaseModel):
    conversation_id: int | None = None
    message: str


class ChatResumeRequest(BaseModel):
    conversation_id: int
    interrupt_id: str
    approved: bool


class ChatResponse(BaseModel):
    conversation_id: int
    answer: str
    status: Literal['completed', 'waiting_approval', 'need_more_info'] = 'completed'
    intent: Literal['knowledge', 'ticket', 'general'] | None = None
    ticket_id: int | None = None
    interrupt_id: str | None = None
    interrupt_value: dict[str, Any] | None = None
