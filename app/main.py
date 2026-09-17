from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.auth import router as auth_router
from app.api.users import router as users_router
from app.api.chat import router as chat_router
from app.api.knowledge import router as knowledge_router
from app.core.redis import create_redis_client
from app.exceptions.base import BusinessException
from app.exceptions.handlers import business_exception_handler


# @asynccontextmanager, 把一个“带资源生命周期的异步生成器”，变成可以用 async with 管理的异步上下文管理器
# 价值：它把“资源创建”和“资源清理”放在同一个函数里，并且让框架自动决定什么时候执行两边。
@asynccontextmanager
async def lifespan(app: FastAPI):
    # fastapi启动前
    redis_client = create_redis_client()
    await redis_client.ping()
    # state 是 Starlette 提供的一个用来保存应用级共享状态/对象的容器
    app.state.redis = redis_client
    # 启动中
    yield
    # 关闭后
    await redis_client.aclose()


app = FastAPI(
    title='AI Agent Backend',
    lifespan=lifespan
)

# 全局异常处理
app.add_exception_handler(
    BusinessException,
    business_exception_handler
)

# 路由注册
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(chat_router)
app.include_router(knowledge_router)
