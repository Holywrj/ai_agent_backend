import httpx
from elasticsearch import AsyncElasticsearch
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import interrupt
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.knowledge import delete_document, get_document
from app.services.reranker import DashScopeReranker
from app.services.retrieval import PgVectorRetriever, ElasticsearchBM25Retriever, HybridRetriever


def create_search_knowledge_tool(
        db: AsyncSession,
        elasticsearch_client: AsyncElasticsearch,
        reranker_client: httpx.AsyncClient,
        candidate_top_k: int = 20,
        final_top_k: int = 5,
        score_threshold: float | None = None
) -> BaseTool:
    vector_retriever = PgVectorRetriever(
        db=db,
        top_k=candidate_top_k,
        score_threshold=score_threshold
    )
    bm25_retriever = ElasticsearchBM25Retriever(
        client=elasticsearch_client,
        top_k=candidate_top_k,
    )
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        top_k=candidate_top_k
    )
    reranker = DashScopeReranker(client=reranker_client)

    @tool
    async def search_knowledge(query: str) -> str:
        """搜索内部知识库，返回经过混合检索和重排序后的相关知识内容。"""
        # hybrid search
        documents = await hybrid_retriever.ainvoke(query)
        if not documents:
            return '知识库中没有找到与问题相关的内容。'
        # rerank
        reranked_documents = await reranker.rerank(
            query=query,
            documents=documents,
            top_n=final_top_k
        )
        if not reranked_documents:
            return '知识库中没有找到与问题相关的内容。'

        results: list[str] = []
        for index, document in enumerate(reranked_documents, start=1):
            results.append(
                f'【知识片段{index}】\n'
                f'{document.page_content}'
            )
        return '\n\n'.join(results)

    return search_knowledge


def create_delete_knowledge_tool(
        db: AsyncSession
) -> BaseTool:

    @tool
    async def delete_knowledge_document(
            document_id: int,
            runtime: ToolRuntime
    ) -> str:
        """
        删除指定的知识库文档。
        当用户明确要求删除知识库文档时，必须调用此工具。
        不要自行向用户询问删除确认。
        工具内部会通过 interrupt() 请求人工审批，
        获得批准后才真正删除文档。
        """
        document = await get_document(
            db=db,
            document_id=document_id
        )
        approval = interrupt(
            {
                'type': 'approval',
                'action': 'delete_knowledge_document',
                'tool_call_id': runtime.tool_call_id,
                'document_id': document.id,
                'title': document.title,
                'message': (
                    f'是否确认删除知识库文档 '
                    f'“{document.title}” (ID: {document.id}) ？'
                )
            }
        )
        if approval is not True:
            return (
                f'删除操作已取消。'
                f'文档“{document.title}”没有被删除'
            )
        await delete_document(
            db=db,
            document_id=document.id
        )
        return (
            f'知识库文档“{document.title}”'
            f' (ID: {document.id}) 已删除'
        )

    return delete_knowledge_document
