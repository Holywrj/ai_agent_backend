from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db, get_current_user
from app.models.user import User
from app.schemas.user import UserCreate, UserResponse
from app.services.user import create_user

router = APIRouter(
    prefix='/users',
    tags=['users']
)


@router.post(
    '',
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED
)
async def register_user(
        user_data: UserCreate,
        db: AsyncSession = Depends(get_db)
):
    return await create_user(db, user_data)


@router.get(
    '/me',
    response_model=UserResponse
)
async def get_me(
        current_user: User = Depends(get_current_user)
):
    return current_user
