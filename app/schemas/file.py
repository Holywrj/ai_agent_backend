from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FileResponse(BaseModel):
    id: int
    original_filename: str
    mime_type: str
    size: int
    sha256: str
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )
