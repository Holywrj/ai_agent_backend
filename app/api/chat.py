from elasticsearch import AsyncElasticsearch
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import httpx
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db, get_redis, get_elasticsearch, get_reranker_client
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse, ChatResumeRequest
from app.services.chat import chat, resume_chat, stream_chat

router = APIRouter(
    prefix='/chat',
    tags=['chat']
)


@router.post(
    '',
    response_model=ChatResponse,
)
async def chat_completion(
        chat_data: ChatRequest,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
        elasticsearch_client: AsyncElasticsearch = Depends(get_elasticsearch),
        reranker_client: httpx.AsyncClient = Depends(get_reranker_client)
):
    result = await chat(
        db=db,
        redis=redis,
        elasticsearch_client=elasticsearch_client,
        reranker_client=reranker_client,
        message=chat_data.message,
        user_id=current_user.id,
        conversation_id=chat_data.conversation_id
    )

    return ChatResponse(
        conversation_id=result.conversation_id,
        answer=result.answer,
        status=result.status,
        intent=result.intent,
        ticket_id=result.ticket_id,
        interrupt_id=result.interrupt_id,
        interrupt_value=result.interrupt_value
    )


@router.post(
    '/resume',
    response_model=ChatResponse,
)
async def resume_chat_completion(
        resume_data: ChatResumeRequest,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
        elasticsearch_client: AsyncElasticsearch = Depends(get_elasticsearch),
        reranker_client: httpx.AsyncClient = Depends(get_reranker_client)
):
    result = await resume_chat(
        db=db,
        redis=redis,
        elasticsearch_client=elasticsearch_client,
        reranker_client=reranker_client,
        conversation_id=resume_data.conversation_id,
        user_id=current_user.id,
        interrupt_id=resume_data.interrupt_id,
        approved=resume_data.approved
    )

    return ChatResponse(
        conversation_id=result.conversation_id,
        answer=result.answer,
        status=result.status,
        intent=result.intent,
        ticket_id=result.ticket_id,
        interrupt_id=result.interrupt_id,
        interrupt_value=result.interrupt_value
    )


@router.post(
    '/stream'
)
async def chat_stream(
        chat_data: ChatRequest,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
        elasticsearch_client: AsyncElasticsearch = Depends(get_elasticsearch),
        reranker_client: httpx.AsyncClient = Depends(get_reranker_client)
):
    # StreamingResponse, 把一个 Python 可迭代的数据流，包装成一个 HTTP 流式响应
    # medis_type -> Content-Type, 'text/event-stream' -> Server-Sent Events（SSE）流
    return StreamingResponse(
        stream_chat(
            db=db,
            redis=redis,
            elasticsearch_client=elasticsearch_client,
            reranker_client=reranker_client,
            message=chat_data.message,
            user_id=current_user.id,
            conversation_id=chat_data.conversation_id
        ),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',  # 告诉nginx，不要缓存此流数据
        }
    )
