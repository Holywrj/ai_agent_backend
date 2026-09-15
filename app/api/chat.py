from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db, get_redis
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import chat

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
    conversation_id, answer = await chat(
        db=db,
        redis=redis,
        message=chat_data.message,
        user_id=current_user.id,
        conversation_id=chat_data.conversation_id
    )

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer
    )
