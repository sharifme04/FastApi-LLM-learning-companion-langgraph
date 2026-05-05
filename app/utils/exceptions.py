"""Global exception handlers."""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger("learning_companion")


class AppError(Exception):
    def __init__(self, message: str, status_code: int = 500, detail: Any = None):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class IngestionError(AppError):
    def __init__(self, message: str, detail: Any = None):
        super().__init__(message, status_code=400, detail=detail)


class RetrievalError(AppError):
    def __init__(self, message: str, detail: Any = None):
        super().__init__(message, status_code=502, detail=detail)


class GenerationError(AppError):
    def __init__(self, message: str, detail: Any = None):
        super().__init__(message, status_code=502, detail=detail)


class CostLimitExceededError(AppError):
    def __init__(self, daily_total: float, limit: float):
        super().__init__(
            f"Daily cost limit exceeded: ${daily_total:.2f} / ${limit:.2f}",
            status_code=429,
            detail={"daily_total": daily_total, "limit": limit},
        )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.error("App error: %s", exc.message, extra={"status": exc.status_code})
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(HTTPException)
    async def http_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.exception_handler(ValidationError)
    async def validation_handler(request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "Validation error", "detail": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", str(exc))
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
