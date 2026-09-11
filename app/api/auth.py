from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db, get_redis, oauth2_scheme
from app.schemas.auth import TokenResponse
from app.services.auth import login, logout

router = APIRouter(
    prefix='/auth',
    tags=['auth']
)


@router.post(
    '/login',
    response_model=TokenResponse
)
async def login_user(
        login_data: OAuth2PasswordRequestForm = Depends(),
        db: AsyncSession = Depends(get_db)
):
    """
    :param login_data: 让 FastAPI 按 OAuth2 Password Flow 接收，而不是 json 的形式
    :param db: 数据库依赖
    :return:
    """
    access_token = await login(
        db,
        login_data.username,
        login_data.password
    )
    return TokenResponse(
        access_token=access_token
    )


@router.post('/logout')
async def logout_user(
        token: str = Depends(oauth2_scheme),
        redis: Redis = Depends(get_redis)
):
    try:
        await logout(token, redis)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Could not validate credentials',
            headers={'WWW-Authenticate': 'Bearer'}
        )
    return {
        'message': 'Logged out successfully'
    }
