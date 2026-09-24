from elasticsearch import AsyncElasticsearch

from app.core.config import settings
from app.core.elasticsearch import ensure_knowledge_index
from app.models.knowledge import KnowledgeChunk


async def index_knowledge_chunk(
        client: AsyncElasticsearch,
        chunk: KnowledgeChunk
) -> None:
    await ensure_knowledge_index(client)
    # wait_for: 等这次写入对搜索可见以后，再返回
    await client.index(
        index=settings.elasticsearch_knowledge_index,
        id=str(chunk.id),
        document={
            'chunk_id': chunk.id,
            'document_id': chunk.document_id,
            'chunk_index': chunk.chunk_index,
            'content': chunk.content,
        },
        refresh='wait_for',
    )


async def search_knowledge_bm25(
        client: AsyncElasticsearch,
        query: str,
        top_k: int = 5
) -> list[dict]:
    response = await client.search(
        index=settings.elasticsearch_knowledge_index,
        # 在 content 这个 text 字段上，对用户输入执行全文匹配，并按照相关性排序
        query={
            'match': {
                'content': query,
            }
        },
        size=top_k,
    )

    return response['hits']['hits']
