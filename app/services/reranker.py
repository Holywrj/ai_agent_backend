import httpx
from langchain_core.documents import Document

from app.core.config import settings


class DashScopeReranker:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def rerank(
            self,
            query: str,
            documents: list[Document],
            top_n: int = 5
    ) -> list[Document]:
        if not documents:
            return []
        payload = {
            'model': settings.dashscope_rerank_model,
            'input': {
                'query': query,
                'documents': [
                    document.page_content
                    for document in documents
                ]
            },
            'parameters': {
                'top_n': min(top_n, len(documents)),
                'return_documents': False
            }
        }
        response = await self.client.post('', json=payload)
        response.raise_for_status()
        data = response.json()
        results = data['output']['results']
        reranked_documents: list[Document] = []
        for rank, result in enumerate(results, start=1):
            original_index = result['index']
            original_document = documents[original_index]
            reranked_documents.append(
                Document(
                    page_content=original_document.page_content,
                    metadata={
                        **original_document.metadata,
                        'rerank_score': result['relevance_score'],
                        'rerank_rank': rank
                    }
                )
            )

        return reranked_documents
