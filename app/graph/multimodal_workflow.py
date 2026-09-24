from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from app.core.checkpointer import checkpointer
from app.core.llm import create_llm
from app.schemas.multimodal import MultimodalTaskType, MultimodalTaskDecision


class MultimodalState(MessagesState):
    """
    Multimodal Workflow 的 State。

    messages:
        LangGraph 当前执行中的消息状态。
    task_type:
        当前多模态业务类型。
    image_url:
        当前需要处理的图片资源地址。
        第一阶段保存资源引用，不直接保存图片二进制。
    image_mime_type:
        图片 MIME 类型，例如 image/png、image/jpeg。
    vision_result:
        Vision Model 产生的视觉理解结果。
    structured_result:
        图片结构化提取后的业务结果。
    knowledge_context:
        visual_knowledge 分支中的知识库检索结果。
    answer:
        当前 Multimodal Workflow 的最终业务结果。
    """
    user_id: int | None
    conversation_id: int | None
    task_type: MultimodalTaskType | None
    image_url: str | None
    image_mime_type: str | None
    vision_result: str | None
    structured_result: dict[str, Any] | None
    knowledge_context: str | None
    answer: str | None


def _latest_human_text(
        state: MultimodalState
) -> str:
    """
    获取当前最新一条HumanMessage的文本内容
    """
    for message in reversed(state['messages']):
        if isinstance(message, HumanMessage):
            if isinstance(message.content, str):
                return message.content

            return str(message.content)

    return ''


def create_multimodal_graph():
    """
    创建独立的Multimodal LangGraph
    """
    router_model = create_llm(thinking=False).with_structured_output(
        MultimodalTaskDecision,
        method='function_calling'
    )

    # todo: 1. Multimodal Router
    async def multimodal_router(state: MultimodalState) -> dict:
        current_message = _latest_human_text(state)
        prompt = [
            SystemMessage(
                content=(
                    '你是多模态业务路由器。'
                    '你只负责判断当前请求应该进入哪个多模态业务分支，'
                    '不要回答用户问题。\n\n'

                    'image_qa：'
                    '用户希望理解、分析、解释图片内容，'
                    '或者询问图片中的报错、现象。\n'

                    'image_extract：'
                    '用户希望从图片中提取明确的结构化字段，'
                    '例如发票、表单、表格信息。\n'

                    'visual_knowledge：'
                    '用户希望先理解图片，'
                    '再结合知识库、操作文档或内部知识回答问题。'
                )
            ),
            HumanMessage(
                content=current_message
            )
        ]
        decision = await router_model.ainvoke(prompt)

        # 每次新的多模态任务开始时，清理上一次Workflow的临时结果
        return {
            'task_type': decision.task_type,
            'vision_result': None,
            'structured_result': None,
            'knowledge_context': None,
            'answer': None,
        }

    def route_multimodal(state: MultimodalState) -> MultimodalTaskType:
        return state.get('task_type') or 'image_qa'

    # todo: 2. image_qa
    async def vision_analyze(state: MultimodalState) -> dict:
        return {
            'vision_result': (
                'TODO：接入 Vision Model 后，'
                '在这里完成图片理解。'
            )
        }

    async def vision_answer(state: MultimodalState) -> dict:
        answer = (
            'TODO：根据 vision_result '
            '生成图片问答结果。'
        )

        return {
            'answer': answer,
            'messages': [
                AIMessage(content=answer)
            ]
        }

    # todo: 3. image_extract
    async def vision_extract(state: MultimodalState) -> dict:
        return {
            'vision_result': (
                'TODO：接入 Vision Model 后，'
                '在这里完成图片内容理解。'
            )
        }

    async def structured_result(state: MultimodalState) -> dict:
        result = {
            'status': 'todo',
            'message': (
                'TODO：使用 Vision Model + '
                'Structured Output 生成结构化结果。'
            )
        }
        answer = 'TODO: 图片结构化提取完成。'

        return {
            'structured_result': result,
            'answer': answer,
            'messages': [
                AIMessage(content=answer)
            ]
        }

    # todo: 4. visual_knowledge
    async def vision_understand(state: MultimodalState) -> dict:
        return {
            'vision_result': (
                'TODO：接入 Vision Model 后，'
                '在这里提取图片语义。'
            )
        }

    async def knowledge_retrieval(state: MultimodalState) -> dict:
        return {
            'knowledge_context': (
                'TODO：使用 vision_result '
                '调用当前 Knowledge Retrieval。'
            )
        }

    async def multimodal_answer(state: MultimodalState) -> dict:
        answer = (
            'TODO：结合图片理解结果和知识库结果 '
            '生成最终答案。'
        )

        return {
            'answer': answer,
            'messages': [
                AIMessage(content=answer)
            ]
        }

    # todo: 5. Build Graph
    builder = StateGraph(MultimodalState)
    # router
    builder.add_node('multimodal_router', multimodal_router)
    # image_qa
    builder.add_node('vision_analyze', vision_analyze)
    builder.add_node('vision_answer', vision_answer)
    # image_extract
    builder.add_node('vision_extract', vision_extract)
    builder.add_node('structured_result', structured_result)
    # visual_knowledge
    builder.add_node('vision_understand', vision_understand)
    builder.add_node('knowledge_retrieval', knowledge_retrieval)
    builder.add_node('multimodal_answer', multimodal_answer)
    # START -> Router
    builder.add_edge(START, 'multimodal_router')
    # router -> 三个多模态业务分支
    builder.add_conditional_edges(
        'multimodal_router',
        route_multimodal,
        {
            'image_qa': 'vision_analyze',
            'image_extract': 'vision_extract',
            'visual_knowledge': 'vision_understand'
        }
    )
    # image_qa
    builder.add_edge('vision_analyze', 'vision_answer')
    builder.add_edge('vision_answer', END)
    # image_extract
    builder.add_edge('vision_extract', 'structured_result')
    builder.add_edge('structured_result', END)
    # visual_knowledge
    builder.add_edge('vision_understand', 'knowledge_retrieval')
    builder.add_edge('knowledge_retrieval', 'multimodal_answer')
    builder.add_edge('multimodal_answer', END)

    return builder.compile(
        checkpointer=checkpointer
    )
