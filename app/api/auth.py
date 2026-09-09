from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.schemas.auth import LoginRequest, TokenResponse
from app.services.auth import login

router = APIRouter(
    prefix='/auth',
    tags=['auth']
)


@router.post('/login', response_model=TokenResponse)
async def login_user(
        login_data: LoginRequest,
        db: AsyncSession = Depends(get_db)
):
    access_token = await login(
        db,
        login_data.username,
        login_data.password
    )
    return TokenResponse(
        access_token=access_token
    )
