from langchain_core.tools import BaseTool, tool
from sqlalchemy.ext.asyncio import AsyncSession

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
