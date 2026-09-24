from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.elasticsearch import ensure_knowledge_index
from app.models.knowledge import KnowledgeChunk


async def rebuild_knowledge_index(
        db: AsyncSession,
        client: AsyncElasticsearch
) -> int:
    """
    将 PostgreSQL 中的全部KnowledgeChunk重建到Elasticsearch
    """
    index_name = settings.elasticsearch_knowledge_index
    # 确保Index存在
    await ensure_knowledge_index(client)
    # 查询Chunk
    result = await db.execute(
        select(KnowledgeChunk)
        .order_by(KnowledgeChunk.id)
    )
    chunks = result.scalars().all()
    if not chunks:
        return 0
    # PostgreSQL -> Elasticsearch Document
    actions = [
        {
            '_index': index_name,
            '_id': str(chunk.id),
            '_source': {
                'chunk_id': chunk.id,
                'document_id': chunk.document_id,
                'chunk_index': chunk.chunk_index,
                'content': chunk.content,
            },
        }
        for chunk in chunks
    ]
    # 批量写入Elasticsearch
    success, errors = await async_bulk(
        client,
        actions,
        refresh='wait_for',
    )

    return success
