import json
from typing import Any, Literal

import httpx
from elasticsearch import AsyncElasticsearch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, default_retry_on, interrupt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.checkpointer import checkpointer
from app.core.llm import create_llm
from app.graph.agent import AgentState, add_agent_branch
from app.graph.multimodal_workflow import create_multimodal_graph
from app.schemas.workflow import IntentDecision, TicketDraftExtraction
from app.services.ticket import create_support_ticket
from app.tools.knowledge import create_search_knowledge_tool
from app.tools.registry import RAG_SCORE_THRESHOLD, RAG_FINAL_TOP_K, RAG_CANDIDATE_TOP_K

WORKFLOW_RETRY_POLICY = RetryPolicy(
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=4.0,
    max_attempts=3,
    jitter=False,
    retry_on=default_retry_on
)


class WorkflowState(AgentState):
    """
    完整业务Workflow State
    """
    intent: Literal['knowledge', 'ticket', 'general', 'multimodal'] | None
    knowledge_context: str | None
    ticket_draft: dict[str, Any] | None
    ticket_approved: bool | None
    ticket_id: int | None
    workflow_status: Literal['completed', 'need_more_info'] | None


def _latest_human_text(
        state: WorkflowState
) -> str:
    for message in reversed(state['messages']):
        if isinstance(message, HumanMessage):
            if isinstance(message.content, str):
                return message.content
            return str(message.content)

    return ''


def _recent_context(
        state: WorkflowState,
        limit: int = 12
):
    return state.get('context_messages', [])[-limit:]


def create_workflow_graph(
        db: AsyncSession,
        elasticsearch_client: AsyncElasticsearch,
        reranker_client: httpx.AsyncClient
):
    router_model = create_llm(thinking=False).with_structured_output(
        IntentDecision,
        method='function_calling'
    )
    ticket_model = create_llm(thinking=False).with_structured_output(
        TicketDraftExtraction,
        method='function_calling'
    )
    answer_model = create_llm()
    multimodal_graph = create_multimodal_graph()

    search_knowledge_tool = create_search_knowledge_tool(
        db=db,
        elasticsearch_client=elasticsearch_client,
        reranker_client=reranker_client,
        candidate_top_k=RAG_CANDIDATE_TOP_K,
        final_top_k=RAG_FINAL_TOP_K,
        score_threshold=RAG_SCORE_THRESHOLD,
    )

    # todo: 1. Intent Router
    async def classify_intent(state: WorkflowState) -> dict:
        attachments = state.get('attachments') or []
        if attachments:
            return {
                'intent': 'multimodal',
                'workflow_status': None,
                'knowledge_context': None,
                'ticket_id': None,
                'ticket_approved': None
            }

        current_message = _latest_human_text(state)
        existing_ticket_draft = state.get('ticket_draft')
        prompt = [
            SystemMessage(
                content=(
                    '你是企业IT助手的意图路由器。'
                    '只负责判断业务意图，不回答用户问题。\n'

                    'knowledge：'
                    '用户想查询内部知识、制度、产品文档、'
                    '操作规范、处理流程。\n'

                    'ticket：'
                    '用户明确要创建、提交、报修、反馈一个IT故障或服务工单。'
                    '如果当前会话已经存在未提交工单草稿，'
                    '用户是在补充工单信息，也选择ticket。\n'

                    'general：'
                    '普通聊天、天气查询、知识库文档删除等，'
                    '需要通用Agent调用工具处理的请求。\n'

                    '不要因为问题内容涉及公司内部就自动选择knowledge。'
                    '只有用户明确在查询/了解/获取知识时才选择knowledge。'
                )
            )
        ]
        if existing_ticket_draft:
            prompt.append(
                SystemMessage(
                    content=(
                        '当前会话存在一个尚未提交的工单草稿。'
                        f'草稿如下：'
                        f'{json.dumps(existing_ticket_draft, ensure_ascii=False)}\n'
                        '如果用户当前消息是在补充这个工单的信息，'
                        '请选择ticket。'
                    )
                )
            )
        prompt.append(
            HumanMessage(
                content=current_message
            )
        )
        decision = await router_model.ainvoke(prompt)

        # 每一轮新请求开始时，清理上一次请求留下的临时结果
        return {
            'intent': decision.intent,
            'workflow_status': None,
            'knowledge_context': None,
            'ticket_id': None,
            'ticket_approved': None,
        }

    def route_intent(state: WorkflowState) -> Literal['knowledge', 'ticket', 'general', 'multimodal']:
        return state.get('intent') or 'general'

    # todo: 2. Knowledge Workflow
    async def search_knowledge_node(state: WorkflowState) -> dict:
        current_message = _latest_human_text(state)
        result = await search_knowledge_tool.ainvoke(
            {
                'query': current_message
            }
        )

        return {
            'knowledge_context': result
        }

    async def answer_knowledge(state: WorkflowState) -> dict:
        current_message = _latest_human_text(state)
        prompt = [
            SystemMessage(
                content=(
                    '你是企业内部知识助手。'
                    '请优先依据下面提供的知识库检索结果回答用户。'
                    '不要编造知识库中不存在的内部规则。'
                    '如果检索结果不足以回答，'
                    '请明确告诉用户目前没有找到足够的内部资料。'
                    '回答简洁、直接。'
                )
            )
        ]
        if state.get('summary'):
            prompt.append(
                SystemMessage(
                    content=(
                        f'会话摘要：\n'
                        f'{state["summary"]}'
                    )
                )
            )
        prompt.extend(_recent_context(state))
        prompt.append(
            SystemMessage(
                content=(
                    '知识库检索结果：\n'
                    f'{state.get("knowledge_context") or "无检索结果"}'
                )
            )
        )
        prompt.append(
            HumanMessage(
                content=current_message
            )
        )
        response = await answer_model.ainvoke(prompt)

        return {
            'messages': [response],
            'workflow_status': 'completed'
        }

    # todo: 3. Ticket Workflow
    async def extract_ticket(state: WorkflowState) -> dict:
        current_message = _latest_human_text(state)
        previous_draft = state.get('ticket_draft')
        prompt = [
            SystemMessage(
                content=(
                    '你是企业IT工单信息提取器。'
                    '根据用户当前消息和已有草稿生成一份完整工单。'
                    '只提取用户明确提供的信息，不要编造。\n'
                    '如果某个字段无法确定，返回空字符串或null。'
                    'category和priority只能根据已有信息判断，'
                    '没有足够信息时不要编造具体原因。'
                )
            )
        ]
        if previous_draft:
            prompt.append(
                SystemMessage(
                    content=(
                        '已有工单草稿如下。'
                        '请根据当前消息继续补充、更新：\n'
                        f'{json.dumps(previous_draft, ensure_ascii=False)}'
                    )
                )
            )
        if state.get('summary'):
            prompt.append(
                SystemMessage(
                    content=(
                        f'会话摘要：\n'
                        f'{state["summary"]}'
                    )
                )
            )
        prompt.extend(_recent_context(state))
        prompt.append(
            HumanMessage(
                content=current_message
            )
        )
        extraction = await ticket_model.ainvoke(prompt)
        draft = extraction.model_dump()
        if not draft.get('category'):
            draft['category'] = 'other'
        if not draft.get('priority'):
            draft['priority'] = 'medium'

        missing_fields: list[str] = []
        if not draft['title'].strip():
            missing_fields.append('title')
        if not draft['description'].strip():
            missing_fields.append('description')
        draft['missing_fields'] = missing_fields
        draft['ready'] = not missing_fields

        return {
            'ticket_draft': draft,
            'ticket_approved': None,
            'ticket_id': None
        }

    def route_ticket(state: WorkflowState) -> Literal['ticket_approval', 'ticket_clarify']:
        draft = state.get('ticket_draft') or {}
        if draft.get('ready'):
            return 'ticket_approval'
        return 'ticket_clarify'

    async def ticket_clarify(state: WorkflowState) -> dict:
        draft = state.get('ticket_draft') or {}
        labels = {
            'title': '问题概述',
            'description': '详细问题描述'
        }
        missing = draft.get('missing_fields', [])
        fields = '、'.join(
            labels.get(item, item)
            for item in missing
        )
        message = (
            '为了帮你创建IT工单，还需要补充：'
            f'{fields}。'
            '请补充后，我会继续为你生成工单。'
        )

        return {
            'messages': [
                AIMessage(
                    content=message
                )
            ],
            'workflow_status': 'need_more_info'
        }

    async def ticket_approval(state: WorkflowState) -> dict:
        draft = state.get('ticket_draft') or {}
        approval = interrupt(
            {
                'type': 'approval',
                'action': 'create_support_ticket',
                'conversation_id': state['conversation_id'],
                'title': draft['title'],
                'description': draft['description'],
                'category': draft['category'],
                'priority': draft['priority'],
                'message': (
                    '即将创建IT工单，'
                    '是否确认提交？'
                )
            }
        )

        return {
            'ticket_approved': approval is True
        }

    def route_ticket_approval(state: WorkflowState) -> Literal['create_ticket', 'cancel_ticket']:
        if state.get('ticket_approved') is True:
            return 'create_ticket'
        return 'cancel_ticket'

    async def create_ticket(state: WorkflowState) -> dict:
        draft = state.get('ticket_draft') or {}
        ticket = await create_support_ticket(
            db=db,
            user_id=state['user_id'],
            conversation_id=state['conversation_id'],
            draft=draft
        )

        return {
            'ticket_id': ticket.id,
            'ticket_draft': None,
            'ticket_approved': None,
            'workflow_status': 'completed',
            'messages': [
                AIMessage(
                    content=(
                        f'IT工单已创建成功，'
                        f'工单编号为 #{ticket.id}。'
                        f'当前状态：{ticket.status}。'
                    )
                )
            ]
        }

    async def cancel_ticket(state: WorkflowState) -> dict:
        return {
            'ticket_id': None,
            'ticket_approved': False,
            'workflow_status': 'completed',
            'messages': [
                AIMessage(
                    content=(
                        '已取消本次IT工单创建，'
                        '未产生新的工单。'
                    )
                )
            ]
        }

    # todo: 4. Build Graph
    builder = StateGraph(WorkflowState)
    builder.add_node('router', classify_intent, retry_policy=WORKFLOW_RETRY_POLICY)
    builder.add_node('knowledge_search', search_knowledge_node)
    builder.add_node('knowledge_answer', answer_knowledge, retry_policy=WORKFLOW_RETRY_POLICY)
    builder.add_node('ticket_extract', extract_ticket, retry_policy=WORKFLOW_RETRY_POLICY)
    builder.add_node('ticket_clarify', ticket_clarify)
    builder.add_node('ticket_approval', ticket_approval)
    builder.add_node('create_ticket', create_ticket)
    builder.add_node('cancel_ticket', cancel_ticket)
    # 注册原来的通用Agent
    add_agent_branch(
        builder,
        db=db,
        elasticsearch_client=elasticsearch_client,
        reranker_client=reranker_client
    )
    # 注册multimodal
    builder.add_node('multimodal', multimodal_graph)
    # START -> Router
    builder.add_edge(START, 'router')
    # Router -> 各个业务分支
    builder.add_conditional_edges(
        'router',
        route_intent,
        {
            'knowledge': 'knowledge_search',
            'ticket': 'ticket_extract',
            'general': 'agent',
            'multimodal': 'multimodal'
        }
    )
    # Knowledge
    builder.add_edge('knowledge_search', 'knowledge_answer')
    builder.add_edge('knowledge_answer', END)
    # Ticket
    builder.add_conditional_edges(
        'ticket_extract',
        route_ticket,
        {
            'ticket_approval': 'ticket_approval',
            'ticket_clarify': 'ticket_clarify'
        }
    )
    builder.add_edge('ticket_clarify', END)
    builder.add_conditional_edges(
        'ticket_approval',
        route_ticket_approval,
        {
            'create_ticket': 'create_ticket',
            'cancel_ticket': 'cancel_ticket'
        }
    )
    builder.add_edge('create_ticket', END)
    builder.add_edge('cancel_ticket', END)
    # multimodal
    builder.add_edge('multimodal', END)

    return builder.compile(
        checkpointer=checkpointer
    )
