from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from langchain_core.documents import Document

from app.core.embedding import create_embeddings
from app.models.knowledge import KnowledgeChunk


async def similarity_search(
        db: AsyncSession,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None
) -> list[Document]:
    embeddings = create_embeddings()

    query_embedding = await embeddings.aembed_query(query)
    distance = KnowledgeChunk.embedding.cosine_distance(query_embedding)
    statement = (
        select(
            KnowledgeChunk,
            distance.label('distance')
        )
        .order_by(distance)
        .limit(top_k)
    )
    result = await db.execute(statement)
    documents: list[Document] = []
    for chunk, chunk_distance in result.all():
        similarity = 1 - float(chunk_distance)
        if score_threshold is not None and similarity < score_threshold:
            continue
        documents.append(
            Document(
                page_content=chunk.content,
                metadata={
                    'document_id': chunk.document_id,
                    'chunk_id': chunk.id,
                    'chunk_index': chunk.chunk_index,
                    'similarity': similarity
                }
            )
        )

    return documents
