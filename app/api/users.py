from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
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
