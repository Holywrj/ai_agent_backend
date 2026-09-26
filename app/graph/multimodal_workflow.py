import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from app.core.checkpointer import checkpointer
from app.core.llm import create_llm
from app.core.multimodal import create_multimodal_llm
from app.core.storage import file_storage
from app.exceptions.base import BusinessException
from app.graph.state import BaseWorkflowState
from app.schemas.multimodal import MultimodalTaskType, MultimodalTaskDecision, MultimodalExtractResult


class MultimodalState(BaseWorkflowState):
    """
    Multimodal Workflow 的 State。

    task_type: 当前多模态业务类型
    vision_result: Vision Model 产生的视觉理解结果
    structured_result: 图片结构化提取后的业务结果
    multimodal_knowledge_context: visual_knowledge 分支中的知识库检索结果
    answer: 当前 Multimodal Workflow 的最终业务结果
    """
    task_type: MultimodalTaskType | None
    vision_result: str | None
    structured_result: dict[str, Any] | None
    multimodal_knowledge_context: str | None
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


async def _build_image_contents(
        state: MultimodalState
) -> list[dict[str, Any]]:
    """
    将当前MultimodalState中的图片附件转换成模型可以使用的image_url内容
    """
    attachments = state.get('attachments') or []
    if not attachments:
        raise BusinessException(
            message='Multimodal attachments are missing',
            code='MULTIMODAL_ATTACHMENTS_MISSING'
        )
    image_contents: list[dict[str, Any]] = []
    for attachment in attachments:
        image_url = await file_storage.get_model_input(
            storage_key=attachment['storage_key'],
            mime_type=attachment['mime_type']
        )
        image_contents.append({
            'type': 'image_url',
            'image_url': {'url': image_url}
        })

    return image_contents


def create_multimodal_graph(
        search_knowledge_tool: BaseTool | None = None
):
    """
    创建独立的Multimodal LangGraph
    """
    router_model = create_llm(thinking=False).with_structured_output(
        MultimodalTaskDecision,
        method='function_calling'
    )
    vision_model = create_multimodal_llm()
    extract_model = create_llm(thinking=False).with_structured_output(
        MultimodalExtractResult,
        method='function_calling'
    )
    answer_model = create_llm()

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
            'multimodal_knowledge_context': None,
            'answer': None,
        }

    def route_multimodal(state: MultimodalState) -> MultimodalTaskType:
        task_type = state.get('task_type')
        if task_type is None:
            raise BusinessException(
                message='Multimodal task_type is missing',
                code='MULTIMODAL_TASK_TYPE_MISSING'
            )

        return task_type

    # todo: 2. image_qa
    async def vision_analyze(state: MultimodalState) -> dict:
        current_message = _latest_human_text(state)
        image_contents = await _build_image_contents(state)
        prompt = [
            SystemMessage(
                content=(
                    '你是一个专业的视觉理解模型。'
                    '请仔细分析用户提供的图片，并结合用户问题提取与问题相关的视觉信息。'
                    '这里只负责视觉分析，不要编造图片中不存在的信息。'
                    '如果无法从图片中确定某项信息，要明确说明无法确定。'
                )
            ),
            HumanMessage(
                content=[
                    {
                        'type': 'text',
                        'text': current_message
                    },
                    *image_contents
                ]
            )
        ]
        response = await vision_model.ainvoke(prompt)
        if isinstance(response.content, str):
            vision_result = response.content
        else:
            vision_result = str(response.content)
        return {
            'vision_result': vision_result
        }

    async def vision_answer(state: MultimodalState) -> dict:
        current_message = _latest_human_text(state)
        vision_result = state.get('vision_result')
        if not vision_result:
            raise BusinessException(
                message='vision_result is missing',
                code='MULTIMODAL_VISION_RESULT_MISSING'
            )
        prompt = [
            SystemMessage(
                content=(
                    '你是一个专业的多模态问答助手。'
                    '请根据用户的问题和视觉模型提供的分析结果回答用户。'
                    '不要虚构视觉模型没有提供的信息。'
                    '回答要准确、直接，必要时说明判断依据。'
                )
            ),
            HumanMessage(
                content=(
                    f'用户问题：\n{current_message}\n\n'
                    f'视觉分析结果：\n{vision_result}'
                )
            ),
        ]
        response = await answer_model.ainvoke(prompt)
        if isinstance(response.content, str):
            answer = response.content
        else:
            answer = str(response.content)

        return {
            'answer': answer,
            'messages': [
                AIMessage(content=answer)
            ]
        }

    # todo: 3. image_extract
    async def vision_extract(state: MultimodalState) -> dict:
        current_message = _latest_human_text(state)
        image_contents = await _build_image_contents(state)
        prompt = [
            SystemMessage(
                content=(
                    '你是一个专业的视觉信息提取模型。'
                    '请仔细阅读用户提供的图片。'
                    '重点识别图片中的文字、字段、编号、日期、金额、名称等可见信息。'
                    '请根据用户的问题关注需要提取的内容。'
                    '这里只负责理解和提取图片信息，不需要输出最终结构化 JSON。'
                    '不要编造图片中不存在的信息。'
                    '无法确认的内容请明确说明。'
                )
            ),
            HumanMessage(
                content=[
                    {
                        'type': 'text',
                        'text': current_message
                    },
                    *image_contents
                ]
            )
        ]
        response = await vision_model.ainvoke(prompt)
        if isinstance(response.content, str):
            vision_result = response.content
        else:
            vision_result = str(response.content)

        return {
            'vision_result': vision_result
        }

    async def structured_result(state: MultimodalState) -> dict:
        current_message = _latest_human_text(state)
        vision_result = state.get('vision_result')
        if not vision_result:
            raise BusinessException(
                message='vision_result is missing',
                code='MULTIMODAL_VISION_RESULT_MISSING'
            )
        prompt = [
            SystemMessage(
                content=(
                    '你是一个结构化信息提取助手。'
                    '请根据用户的问题和视觉模型提供的分析结果，'
                    '提取用户明确要求的信息。'
                    '只输出符合结构要求的数据。'
                    '不要补充视觉分析中不存在的信息。'
                    '所有字段值使用字符串表示。'
                )
            ),
            HumanMessage(
                content=(
                    f'用户要求：\n{current_message}\n\n'
                    f'视觉分析结果：\n{vision_result}'
                )
            )
        ]
        result = await extract_model.ainvoke(prompt)
        if not isinstance(result, MultimodalExtractResult):
            raise BusinessException(
                message='Structured extraction result is invalid',
                code='MULTIMODAL_STRUCTURED_RESULT_INVALID'
            )
        data = result.data
        # indent，每层缩进空格数
        answer = json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        )

        return {
            'structured_result': data,
            'answer': answer,
            'messages': [
                AIMessage(content=answer)
            ]
        }

    # todo: 4. visual_knowledge
    async def vision_understand(state: MultimodalState) -> dict:
        current_message = _latest_human_text(state)
        image_contents = await _build_image_contents(state)
        prompt = [
            SystemMessage(
                content=(
                    '你是一个专业的视觉理解模型。'
                    '请分析用户提供的图片，并提取与知识库检索相关的视觉信息。'
                    '重点关注图片中的文字、报错信息、产品名称、功能名称、'
                    '界面元素、设备信息以及其他能够帮助定位知识内容的关键信息。'
                    '这里只负责理解图片，不要直接回答用户问题。'
                    '不要编造图片中不存在的信息。'
                    '无法确认的内容请明确说明。'
                )
            ),
            HumanMessage(
                content=[
                    {
                        'type': 'text',
                        'text': current_message
                    },
                    *image_contents
                ]
            )
        ]
        response = await vision_model.ainvoke(prompt)
        if isinstance(response.content, str):
            vision_result = response.content
        else:
            vision_result = str(response.content)

        return {
            'vision_result': vision_result
        }

    async def knowledge_retrieval(state: MultimodalState) -> dict:
        if search_knowledge_tool is None:
            raise BusinessException(
                message='Knowledge retrieval tool is not configured',
                code='MULTIMODAL_KNOWLEDGE_TOOL_MISSING'
            )
        current_message = _latest_human_text(state)
        vision_result = state.get('vision_result')
        if not vision_result:
            raise BusinessException(
                message='vision_result is missing',
                code='MULTIMODAL_VISION_RESULT_MISSING'
            )
        query = (
            f'用户问题：\n{current_message}\n\n'
            f'图片视觉信息：\n{vision_result}'
        )
        result = await search_knowledge_tool.ainvoke(
            {
                'query': query
            }
        )

        return {
            'multimodal_knowledge_context': result
        }

    async def multimodal_answer(state: MultimodalState) -> dict:
        current_message = _latest_human_text(state)
        vision_result = state.get('vision_result')
        knowledge_context = state.get('multimodal_knowledge_context')
        if not vision_result:
            raise BusinessException(
                message='vision_result is missing',
                code='MULTIMODAL_VISION_RESULT_MISSING'
            )
        if not knowledge_context:
            raise BusinessException(
                message='multimodal_knowledge_context is missing',
                code='MULTIMODAL_KNOWLEDGE_CONTEXT_MISSING'
            )
        prompt = [
            SystemMessage(
                content=(
                    '你是企业内部多模态知识助手。'
                    '请结合用户问题、图片视觉理解结果和知识库检索结果回答问题。\n\n'

                    '回答规则：'
                    '1. 知识库检索结果是回答企业内部问题的主要事实依据。'
                    '2. 涉及企业内部制度、流程、产品、操作规范的问题，'
                    '只使用知识库中明确提供的信息，不要自行补充内部规则。'
                    '3. 如果知识库结果不足以回答问题，'
                    '明确说明当前知识库没有找到足够的信息，不要编造答案。'
                    '4. 图片视觉结果只用于理解当前图片中的现象、文字和上下文，'
                    '不能把视觉模型没有识别到的信息当成事实。'
                    '5. 可以使用通用常识帮助解释问题，但不要把通用常识伪装成企业内部知识。'
                    '6. 回答要直接、清晰，并尽量针对用户实际问题给出处理建议。'
                )
            ),
            HumanMessage(
                content=(
                    f'用户问题：\n{current_message}\n\n'
                    f'图片视觉理解：\n{vision_result}\n\n'
                    f'知识库检索结果：\n{knowledge_context}'
                )
            )
        ]
        response = await answer_model.ainvoke(prompt)
        if isinstance(response.content, str):
            answer = response.content
        else:
            answer = str(response.content)

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
