from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db, get_redis
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse, ChatResumeRequest
from app.services.chat import chat, resume_chat

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
        redis: Redis = Depends(get_redis)
):
    result = await chat(
        db=db,
        redis=redis,
        message=chat_data.message,
        user_id=current_user.id,
        conversation_id=chat_data.conversation_id
    )

    return ChatResponse(
        conversation_id=result.conversation_id,
        answer=result.answer,
        status=result.status,
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
        redis: Redis = Depends(get_redis)
):
    result = await resume_chat(
        db=db,
        redis=redis,
        conversation_id=resume_data.conversation_id,
        user_id=current_user.id,
        interrupt_id=resume_data.interrupt_id,
        approved=resume_data.approved
    )

    return ChatResponse(
        conversation_id=result.conversation_id,
        answer=result.answer,
        status=result.status,
        interrupt_id=result.interrupt_id,
        interrupt_value=result.interrupt_value
    )
