from langchain_openai import ChatOpenAI

from app.core.config import settings


def create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.deepseek_chat_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url
    )
