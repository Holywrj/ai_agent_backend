from fastapi import FastAPI

from app.api.auth import router as auth_router
from app.api.users import router as users_router
from app.exceptions.base import BusinessException
from app.exceptions.handlers import business_exception_handler

app = FastAPI(
    title='AI Agent Backend'
)

# 全局异常处理
app.add_exception_handler(
    BusinessException,
    business_exception_handler
)

# 路由注册
app.include_router(auth_router)
app.include_router(users_router)
