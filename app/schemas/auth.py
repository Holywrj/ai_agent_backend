from pydantic import BaseModel


# 使用swagger进行登录，LoginRequest 可以删掉，因为 OAuth2 标准表单会由 FastAPI 提供
# class LoginRequest(BaseModel):
#     username: str
#     password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'
