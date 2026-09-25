from langchain_openai import ChatOpenAI

from app.core.config import settings


def create_multimodal_llm() -> ChatOpenAI:
    """
    创建多模态模型。
    """
    return ChatOpenAI(
        model=settings.dashscope_omni_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_omni_base_url
    )
