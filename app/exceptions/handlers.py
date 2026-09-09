from fastapi import Request
from fastapi.responses import JSONResponse

from app.exceptions.base import BusinessException


async def business_exception_handler(
        request: Request,
        exc: BusinessException
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            'code': exc.code,
            'message': exc.message
        }
    )
