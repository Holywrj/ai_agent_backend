from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


def create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.dashscope_embedding_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url
    )
