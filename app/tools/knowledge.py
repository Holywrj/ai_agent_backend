from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import interrupt
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.knowledge import delete_document, get_document
from app.services.retrieval import PgVectorRetriever


def create_search_knowledge_tool(
        db: AsyncSession,
        top_k: int = 5,
        score_threshold: float | None = None
) -> BaseTool:
    retriever = PgVectorRetriever(
        db=db,
        top_k=top_k,
        score_threshold=score_threshold
    )

    @tool
    async def search_knowledge(query: str) -> str:
        """搜索内部知识库，返回与用户问题最相关的知识内容。"""
        documents = await retriever.ainvoke(query)
        if not documents:
            return '知识库中没有找到与问题相关的内容。'
        results: list[str] = []
        for index, document in enumerate(documents, start=1):
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
