from datetime import datetime, timezone

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.jwt import create_access_token, decode_access_token
from app.core.security import verify_password
from app.exceptions.base import BusinessException
from app.models.user import User


async def login(
        db: AsyncSession,
        username: str,
        password: str
) -> str:
    result = await db.execute(
        select(User).where(User.username == username)
    )

    user = result.scalar_one_or_none()
    if not user:
        raise BusinessException(
            message='Incorrect username or password',
            code='INVALID_CREDENTIALS'
        )
    if not verify_password(password, user.hashed_password):
        raise BusinessException(
            message='Incorrect username or password',
            code='INVALID_CREDENTIALS'
        )

    return create_access_token(user.id)


async def logout(
        token: str,
        redis: Redis
) -> None:
    payload = decode_access_token(token)

    exp = payload.get('exp')
    jti = payload.get('jti')
    if exp is None or jti is None:
        raise ValueError('Invalid token payload')
    remaining_seconds = int(exp - datetime.now(timezone.utc).timestamp())

    if remaining_seconds <= 0:
        # Token 已经自然过期, jwt 已无效
        return

    revoked_key = f'revoked:{jti}'
    await redis.set(
        revoked_key,
        '1',
        ex=remaining_seconds
    )
