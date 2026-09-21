from typing import Literal

from pydantic import BaseModel, Field


class IntentDecision(BaseModel):
    """
    Router 节点的输出结构
    """
    intent: Literal['knowledge', 'ticket', 'general'] = Field(
        description=(
            '用户请求的业务意图。'
            'knowledge=内部知识查询；'
            'ticket=创建/提交IT工单；'
            'general=普通对话或需要通用Agent处理的请求。'
        )
    )


class TicketDraftExtraction(BaseModel):
    title: str = Field(
        description=(
            '工单标题。'
            '无法从用户信息确定时返回空字符串，不要编造。'
        )
    )
    description: str = Field(
        description=(
            '工单详细问题描述。'
            '无法从用户信息确定时返回空字符串，不要编造。'
        )
    )
    category: Literal['hardware', 'software', 'network', 'account', 'other'] | None = Field(
        default=None,
        description=('工单类型。')
    )
    priority: Literal['low', 'medium', 'high'] | None = Field(
        default=None,
        description=('工单优先级。')
    )
