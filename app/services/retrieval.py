import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from elasticsearch import AsyncElasticsearch

from app.core.embedding import create_embeddings
from app.core.config import settings
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


class ElasticsearchBM25Retriever(BaseRetriever):
    client: AsyncElasticsearch = Field(exclude=True)
    top_k: int = 5
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
            'ElasticsearchBM25Retriever only supports async retrieval.'
        )

    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager=None
    ) -> list[Document]:
        # 在 content 这个 text 字段上，对用户输入执行全文匹配，并按照相关性排序
        response = await self.client.search(
            index=settings.elasticsearch_knowledge_index,
            query={
                'match': {
                    'content': query
                }
            },
            size=self.top_k
        )
        documents: list[Document] = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            documents.append(
                Document(
                    page_content=source['content'],
                    metadata={
                        'document_id': source['document_id'],
                        'chunk_id': source['chunk_id'],
                        'chunk_index': source['chunk_index'],
                        'bm25_score': hit['_score']
                    }
                )
            )

        return documents


class HybridRetriever(BaseRetriever):
    vector_retriever: PgVectorRetriever = Field(exclude=True)
    bm25_retriever: ElasticsearchBM25Retriever = Field(exclude=True)
    top_k: int = 5
    rrf_k: int = 60
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
            'HybridRetriever only supports async retrieval.'
        )

    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager=None
    ) -> list[Document]:
        vector_documents, bm25_documents = await asyncio.gather(
            self.vector_retriever.ainvoke(query),
            self.bm25_retriever.ainvoke(query)
        )
        scores: dict[int, float] = {}
        documents: dict[int, Document] = {}
        metadata: dict[int, dict] = {}
        for rank, document in enumerate(vector_documents, start=1):
            chunk_id = document.metadata['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (self.rrf_k + rank)
            documents[chunk_id] = document
            metadata.setdefault(chunk_id, {}).update(document.metadata)
            metadata[chunk_id]['vector_rank'] = rank
        for rank, document in enumerate(bm25_documents, start=1):
            chunk_id = document.metadata['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (self.rrf_k + rank)
            documents.setdefault(chunk_id, document)
            metadata.setdefault(chunk_id, {}).update(document.metadata)
            metadata[chunk_id]['bm25_rank'] = rank
        ranked_chunk_ids = sorted(scores, key=scores.get, reverse=True)[:self.top_k]
        results: list[Document] = []
        for chunk_id in ranked_chunk_ids:
            results.append(
                Document(
                    page_content=documents[chunk_id].page_content,
                    metadata={
                        **metadata[chunk_id],
                        'rrf_score': scores[chunk_id]
                    }
                )
            )

        return results
