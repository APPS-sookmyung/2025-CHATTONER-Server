"""
FastAPI Exception Handler Configuration
Global exception handling and error response formatting
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

logger = logging.getLogger(__name__)


def setup_exception_handlers(app: FastAPI):
    """Setup exception handlers"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP exception handler"""
        print(f"[EXCEPTION] HTTP Exception: {exc.status_code} - {exc.detail}")
        print(f"[EXCEPTION] Request URL: {request.url}")
        logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail} for {request.url}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Request validation exception handler"""
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "details": exc.errors()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """General exception handler"""
        logger.error(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred"
            }
        )