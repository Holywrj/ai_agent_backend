from pydantic import BaseModel, ConfigDict, EmailStr


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr

    # ConfigDict 相比 Dict，让配置具有更好的类型提示和 IDE 支持
    model_config = ConfigDict(from_attributes=True)
