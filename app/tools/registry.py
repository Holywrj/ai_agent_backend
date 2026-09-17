from langchain_core.tools import BaseTool
from sqlalchemy.ext.asyncio import AsyncSession

from app.tools.weather import get_weather
from app.tools.knowledge import create_search_knowledge_tool

RAG_TOP_K = 5
RAG_SCORE_THRESHOLD = None


def get_all_tools(
        db: AsyncSession
) -> list[BaseTool]:
    return [
        get_weather,
        create_search_knowledge_tool(
            db=db,
            top_k=RAG_TOP_K,
            score_threshold=RAG_SCORE_THRESHOLD
        ),
    ]
