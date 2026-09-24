from typing import Literal

from pydantic import BaseModel, Field

MultimodalTaskType = Literal[
    'image_qa',
    'image_extract',
    'visual_knowledge',
]


class MultimodalTaskDecision(BaseModel):
    task_type: MultimodalTaskType = Field(
        description=(
            '多模态任务类型。'
            'image_qa=图片问答/图片分析；'
            'image_extract=从图片中提取结构化信息；'
            'visual_knowledge=根据图片理解结果继续查询知识库。'
        )
    )
