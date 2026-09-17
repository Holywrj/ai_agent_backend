from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.exceptions.base import BusinessException
from app.models.user import User
from app.schemas.knowledge import KnowledgeDocumentCreate, KnowledgeDocumentResponse
from app.services.knowledge import delete_document, get_document, ingest_document

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
