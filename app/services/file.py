from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.storage import file_storage
from app.exceptions.base import BusinessException
from app.exceptions.storage import FileTooLargeError
from app.models.file import File

ALLOWED_IMAGE_MIME_TYPES = {
    'image/png',
    'image/jpeg',
    'image/webp',
}


async def save_uploaded_file(
        db: AsyncSession,
        upload_file: UploadFile,
        user_id: int
) -> File:
    """
    保存用户上传的文件，并创建数据库记录。
    """
    filename = Path(upload_file.filename or '').name
    if not filename:
        raise BusinessException(
            message='Filename is required',
            code='FILE_NAME_MISSING'
        )
    mime_type = upload_file.content_type or ''
    if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise BusinessException(
            message='Only PNG, JPEG and WebP images are allowed',
            code='UNSUPPORTED_FILE_TYPE'
        )
    storage_key = f'{user_id}/{uuid4().hex}'
    try:
        # 保存文件
        size, sha256 = await file_storage.save(
            upload_file.file,
            storage_key,
            settings.file_max_size
        )
    except FileTooLargeError:
        raise BusinessException(
            message='File size exceeds the maximum allowed size',
            code='FILE_TOO_LARGE'
        )
    file_record = File(
        user_id=user_id,
        original_filename=filename[:255],
        storage_key=storage_key,
        mime_type=mime_type,
        size=size,
        sha256=sha256
    )
    try:
        db.add(file_record)
        await db.commit()
        await db.refresh(file_record)
    except Exception:
        await db.rollback()
        await file_storage.delete(storage_key)
        raise

    return file_record


async def get_user_file(
        db: AsyncSession,
        file_id: int,
        user_id: int
) -> File:
    """
    获取当前用户自己的文件。
    """
    result = await db.execute(
        select(File)
        .where(
            File.id == file_id,
            File.user_id == user_id
        )
    )
    file_record = result.scalar_one_or_none()
    if file_record is None:
        raise BusinessException(
            message='File not found',
            code='FILE_NOT_FOUND'
        )

    return file_record
