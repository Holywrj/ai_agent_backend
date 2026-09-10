from collections.abc import AsyncGenerator

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from app.core.database import AsyncSessionLocal
from app.core.jwt import decode_access_token
from app.models.user import User

# 从请求的 Authorization Header 中拿出 Bearer Token
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl='/auth/login'
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


async def get_current_user(
        token: str = Depends(oauth2_scheme),
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis)
) -> User:
    try:
        payload = decode_access_token(token)

        user_id = payload.get('sub')
        jti = payload.get('jti')
        if user_id is None:
            raise ValueError('Missing sub')
        if jti is None:
            raise ValueError('Missing jti')

        user_id = int(user_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Could not validate credentials',
            headers={'WWW-Authenticate': 'Bearer'}
        )

    # 检查用户是否注销，token已被销毁并存入redis
    revoked_key = f'revoked:{jti}'
    if await redis.exists(revoked_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Token has been revoked',
            headers={'WWW-Authenticate': 'Bearer'}
        )

    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Could not validate credentials',
            headers={'WWW-Authenticate': 'Bearer'}
        )
    return user
