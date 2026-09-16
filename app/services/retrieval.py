from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from app.core.embedding import create_embeddings
from app.models.knowledge import KnowledgeChunk


class PgVectorRetriever(BaseRetriever):
    # 运行时资源
    # db 在需要运行时使用，但不要把它当作可序列化配置的一部分
    db: AsyncSession = Field(exclude=True)
    # 配置型字段
    top_k: int = 5
    score_threshold: float | None = None  # 最低相似度阈值

    # arbitrary_types_allowed=True
    # 允许 Model 中出现这种不是 Pydantic 原生模型的任意 Python 类型
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager=None
    ) -> list[Document]:
        raise NotImplementedError(
            'PgVectorRetriever only supports async retrieval.'
        )

    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager=None
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
            .limit(self.top_k)
        )
        result = await self.db.execute(statement)
        documents: list[Document] = []
        for chunk, chunk_distance in result.all():
            similarity = 1 - float(chunk_distance)
            if self.score_threshold is not None and similarity < self.score_threshold:
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
