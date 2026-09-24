from elasticsearch import AsyncElasticsearch
from langchain_core.tools import BaseTool
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from app.tools.weather import get_weather
from app.tools.knowledge import create_search_knowledge_tool, create_delete_knowledge_tool

RAG_CANDIDATE_TOP_K = 20
RAG_FINAL_TOP_K = 5
RAG_SCORE_THRESHOLD = None


def get_all_tools(
        db: AsyncSession,
        elasticsearch_client: AsyncElasticsearch,
        reranker_client: httpx.AsyncClient,
) -> list[BaseTool]:
    return [
        get_weather,
        create_search_knowledge_tool(
            db=db,
            elasticsearch_client=elasticsearch_client,
            reranker_client=reranker_client,
            candidate_top_k=RAG_CANDIDATE_TOP_K,
            final_top_k=RAG_FINAL_TOP_K,
            score_threshold=RAG_SCORE_THRESHOLD
        ),
        create_delete_knowledge_tool(
            db=db
        ),
    ]
