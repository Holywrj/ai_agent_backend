from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.exceptions.base import BusinessException
from app.models.user import User
from app.schemas.user import UserCreate


async def create_user(
        db: AsyncSession,
        user_data: UserCreate
) -> User:
    result = await db.execute(
        select(User).where(
            (User.username == user_data.username)
            | (User.email == user_data.email)
        )
    )
    existing_user = result.scalar_one_or_none()
    if existing_user:
        raise BusinessException(
            message='Username or email already exists',
            code='USER_ALREADY_EXISTS'
        )
    user = User(
        username=user_data.username,
        email=str(user_data.email),
        hashed_password=hash_password(user_data.password)
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user
