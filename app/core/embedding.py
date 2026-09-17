from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


def create_embeddings() -> OpenAIEmbeddings:
    # check_embedding_ctx_length: 不让langchain先把文本token化
    # dashscope收到原始str / list[str]
    # chunk_size=10， 一次最多发送10个chunk，符合dashscope_embedding_model的批量限制
    return OpenAIEmbeddings(
        model=settings.dashscope_embedding_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        check_embedding_ctx_length=False,
        chunk_size=10
    )
