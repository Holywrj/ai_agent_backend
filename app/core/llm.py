from langchain_openai import ChatOpenAI

from app.core.config import settings


def create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model='deepseek-v4-pro',
        api_key=settings.deepseek_api_key,
        base_url='https://api.deepseek.com'
    )
