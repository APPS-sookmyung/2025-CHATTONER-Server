"""
FastAPI Middleware Configuration
Middleware setup for CORS, logging, error handling, etc.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# New logging middleware function
async def log_requests_middleware(request: Request, call_next):
    """
    Middleware for logging API requests and response bodies
    """
    logger = logging.getLogger("api.access")
    
    # --- Request information logging ---
    logger.info(f"Request: {request.method} {request.url}")
    
    # --- Response processing ---
    response = await call_next(request)
    
    # Consume stream to read response body.
    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk
    
    # --- Response information logging ---
    log_message = f"Response: {response.status_code}"
    try:
        # Pretty print if response body is in JSON format.
        response_json = json.loads(response_body)
        log_message += f"\n{json.dumps(response_json, indent=2, ensure_ascii=False)}"
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Output as text if not JSON.
        log_message += f" Body: {response_body.decode(errors='ignore')}"

    logger.info(log_message)

    # Since stream was consumed, create and return new response with same content.
    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type
    )


def setup_middleware(app: FastAPI, settings):
    """Setup middleware"""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # --- Logging configuration (modified part) ---
    if settings.DEBUG:
        # Configure "api.access" logger directly to separate from uvicorn logger.
        logger = logging.getLogger("api.access")
        logger.setLevel(logging.INFO)
        
        # Handler and formatter configuration
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Remove existing handlers and add new ones to prevent duplicate logging.
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(handler)
        
        # Prevent propagation to root logger for independent operation.
        logger.propagate = False

    # Add logging middleware only in debug mode
    if settings.DEBUG:
        app.add_middleware(BaseHTTPMiddleware, dispatch=log_requests_middleware)
