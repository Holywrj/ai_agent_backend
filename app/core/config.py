from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = 'AI Agent Backend'
    app_env: str = 'development'

    database_url: str
    redis_url: str

    jwt_secret_key: str
    jwt_algorithm: str = 'HS256'
    jwt_access_token_expire_minutes: int = 30

    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_chat_model: str

    dashscope_api_key: str
    dashscope_base_url: str
    dashscope_embedding_model: str
    dashscope_embedding_dimensions: int

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )


settings = Settings()
