from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.embedding import create_embeddings
from app.models.knowledge import KnowledgeDocument, KnowledgeChunk
from app.exceptions.base import BusinessException

# 切分字符长度
DEFAULT_CHUNK_SIZE = 800
# chunk间共享字符长度
DEFAULT_CHUNK_OVERLAP = 120


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    创建一个“文本切分器”对象，后面用它把一篇很长的文档切成多个 Chunk。
    :return: 递归字符切分器
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        separators=[
            '\n\n',
            '\n',
            '。',
            '！',
            '？',
            '；',
            '.',
            '!',
            '?',
            ';',
            ' ',
            '',
        ]
    )


def split_document(
        title: str,
        content: str,
        source: str | None = None
) -> list[Document]:
    """
    把长文本切成多个 Chunk。
    把每个 Chunk 包装成 LangChain Document，并附上 metadata.
    """
    splitter = create_text_splitter()

    # 切出的chunk会继承metadata
    return splitter.create_documents(
        texts=[content],
        metadatas=[
            {
                'title': title,
                'source': source,
            }
        ]
    )


async def create_knowledge_document(
        db: AsyncSession,
        title: str,
        source: str | None = None
) -> KnowledgeDocument:
    document = KnowledgeDocument(
        title=title,
        source=source
    )
    db.add(document)
    # 先拿到document.id, 但此时还不提交事务
    # 把当前 Session 中待执行的 SQL 刷到数据库，让数据库先执行，但不提交事务
    await db.flush()

    return document


async def save_document_chunks(
        db: AsyncSession,
        document: KnowledgeDocument,
        chunks: list[Document]
) -> list[KnowledgeChunk]:
    embeddings = create_embeddings()
    texts = [
        chunk.page_content
        for chunk in chunks
    ]
    vectors = await embeddings.aembed_documents(texts)
    knowledge_chunks = [
        KnowledgeChunk(
            document_id=document.id,
            chunk_index=index,
            content=chunk.page_content,
            embedding=vector
        )
        for index, (chunk, vector) in enumerate(
            zip(chunks, vectors, strict=True)
        )
    ]
    db.add_all(knowledge_chunks)

    return knowledge_chunks


async def ingest_document(
        db: AsyncSession,
        title: str,
        content: str,
        source: str | None = None
) -> KnowledgeDocument:
    chunks = split_document(
        title=title,
        content=content,
        source=source
    )
    if not chunks:
        raise BusinessException(
            message='Document content cannot be empty',
            code='DOCUMENT_CONTENT_EMPTY'
        )

    document = await create_knowledge_document(
        db=db,
        title=title,
        source=source
    )
    await save_document_chunks(
        db=db,
        document=document,
        chunks=chunks
    )
    await db.commit()
    await db.refresh(document)

    return document


async def get_document(
        db: AsyncSession,
        document_id: int
) -> KnowledgeDocument:
    result = await db.execute(
        select(KnowledgeDocument).where(
            KnowledgeDocument.id == document_id
        )
    )

    document = result.scalar_one_or_none()
    if document is None:
        raise BusinessException(
            message='knowledge Document not found',
            code='KNOWLEDGE_DOCUMENT_NOT_FOUND'
        )

    return document


async def delete_document(
        db: AsyncSession,
        document_id: int
) -> None:
    await db.execute(
        delete(KnowledgeDocument).where(
            KnowledgeDocument.id == document_id
        )
    )
    await db.commit()
