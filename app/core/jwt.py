"""
    JWT三部分组成：Header.Payload.Signature
        Header 告诉服务器使用什么算法
        Payload 数据
        Signature 服务器利用SECRET_KEY对Header + Payload进行签名
"""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
from jwt.exceptions import InvalidTokenError

from app.core.config import settings


def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    payload = {
        'sub': str(user_id),
        'jti': str(uuid4()),  # 唯一标识一个token
        'exp': expire
    }
    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
    except InvalidTokenError:
        raise ValueError('Invalid token')
