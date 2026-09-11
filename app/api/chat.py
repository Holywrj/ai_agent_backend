from fastapi import APIRouter, Depends

from app.api.dependencies import get_current_user
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
        current_user: User = Depends(get_current_user)
):
    answer = await chat(
        message=chat_data.message,
        user_id=current_user.id
    )

    return ChatResponse(
        answer=answer
    )
