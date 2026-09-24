from elasticsearch import AsyncElasticsearch

from app.core.config import settings


def create_elasticsearch_client() -> AsyncElasticsearch:
    return AsyncElasticsearch(
        settings.elasticsearch_url
    )


async def ensure_knowledge_index(
        client: AsyncElasticsearch
) -> None:
    index_name = settings.elasticsearch_knowledge_index
    exists = await client.indices.exists(index=index_name)
    if exists:
        return
    await client.indices.create(
        index=index_name,
        mappings={
            'properties': {
                'content': {'type': 'text'},
                'document_id': {'type': 'integer'},
                'chunk_id': {'type': 'integer'},
                'chunk_index': {'type': 'integer'},
            }
        },
    )
