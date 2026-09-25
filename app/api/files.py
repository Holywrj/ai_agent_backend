from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.models.user import User
from app.schemas.file import FileResponse
from app.services.file import save_uploaded_file

router = APIRouter(
    prefix='/files',
    tags=['files']
)


@router.post(
    '/upload',
    response_model=FileResponse,
    status_code=status.HTTP_201_CREATED
)
async def upload_file(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)
):
    """
    :param file: Ellipsis, 参数必须提供
    :param current_user:
    :param db:
    :return:
    """
    try:
        return await save_uploaded_file(
            db=db,
            upload_file=file,
            user_id=current_user.id,
        )
    finally:
        await file.close()
