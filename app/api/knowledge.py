from pathlib import Path
from typing import Literal

from elasticsearch import AsyncElasticsearch
from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db, get_elasticsearch
from app.exceptions.base import BusinessException
from app.models.user import User
from app.schemas.knowledge import KnowledgeDocumentCreate, KnowledgeDocumentResponse
from app.services.elasticsearch import rebuild_knowledge_index
from app.services.knowledge import delete_document, get_document, ingest_document
from app.services.retrieval import PgVectorRetriever, ElasticsearchBM25Retriever, HybridRetriever

router = APIRouter(
    prefix='/knowledge/documents',
    tags=['knowledge']
)

ALLOWED_FILE_EXTENSIONS = {'.txt', '.md'}
MAX_FILE_SIZE = 5 * 1024 * 1024


@router.post(
    '',
    response_model=KnowledgeDocumentResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_knowledge_document(
        document_data: KnowledgeDocumentCreate,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    return await ingest_document(
        db=db,
        title=document_data.title,
        content=document_data.content,
        source=document_data.source
    )


@router.post(
    '/upload',
    response_model=KnowledgeDocumentResponse,
    status_code=status.HTTP_201_CREATED
)
async def upload_knowledge_document(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    filename = file.filename or ''
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_FILE_EXTENSIONS:
        raise BusinessException(
            message='Only .txt and .md files are allowed',
            code='UNSUPPORTED_FILE_TYPE'
        )
    content_bytes = await file.read()
    try:
        if len(content_bytes) > MAX_FILE_SIZE:
            raise BusinessException(
                message='File size must not exceed 5MB',
                code='FILE_TOO_LARGE'
            )
        try:
            content = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            raise BusinessException(
                message='File must be encoded in UTF-8',
                code='INVALID_FILE_ENCODING'
            )
        if not content.strip():
            raise BusinessException(
                message='Document must not be empty',
                code='DOCUMENT_CONTENT_EMPTY'
            )
        title = Path(filename).stem

        return await ingest_document(
            db=db,
            title=title,
            content=content,
            source=filename
        )
    finally:
        await file.close()


@router.post('/reindex')
async def reindex_knowledge(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        client: AsyncElasticsearch = Depends(get_elasticsearch)
):
    count = await rebuild_knowledge_index(
        db=db,
        client=client
    )

    return {
        'indexed_chunks': count
    }


@router.get('/search')
async def search_knowledge(
        query: str,
        mode: Literal['vector', 'bm25', 'hybrid'] = Query('vector'),
        top_k: int = Query(5, ge=1, le=20),
        db: AsyncSession = Depends(get_db),
        client: AsyncElasticsearch = Depends(get_elasticsearch)
):
    if mode == 'vector':
        retriever = PgVectorRetriever(
            db=db,
            top_k=top_k
        )
    elif mode == 'bm25':
        retriever = ElasticsearchBM25Retriever(
            client=client,
            top_k=top_k
        )
    else:
        vector_retriever = PgVectorRetriever(
            db=db,
            top_k=top_k
        )
        bm25_retriever = ElasticsearchBM25Retriever(
            client=client,
            top_k=top_k
        )
        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            top_k=top_k
        )
    documents = await retriever.ainvoke(query)
    return {
        'query': query,
        'mode': mode,
        'results': [
            {
                'content': document.page_content,
                'metadata': document.metadata,
            }
            for document in documents
        ]
    }


@router.get(
    '/{document_id}',
    response_model=KnowledgeDocumentResponse
)
async def get_knowledge_document(
        document_id: int,
        db: AsyncSession = Depends(get_db)
):
    return await get_document(
        db=db,
        document_id=document_id
    )


@router.delete(
    '/{document_id}',
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_knowledge_document(
        document_id: int,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    await delete_document(
        db=db,
        document_id=document_id
    )
