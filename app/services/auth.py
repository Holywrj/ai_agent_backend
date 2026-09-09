from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.jwt import create_access_token
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
