from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

# Session 是实际进行数据库操作的工作单元
# echo=True, SQLAlchemy 是否把执行的 SQL 打印到终端
engine = create_async_engine(
    settings.database_url,
    echo=True
)

# expire_on_commit=False, Session commit() 以后，不要把已经加载到 Python 对象里的数据自动标记为过期
# 在异步 SQLAlchemy 中，这个配置很常见，也能避免一些对象在 commit 后被访问时触发隐式数据库操作的问题。
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


class Base(DeclarativeBase):
    pass
