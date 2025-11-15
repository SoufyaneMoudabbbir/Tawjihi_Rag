"""
Middleware for FastAPI application
Includes rate limiting, CORS, and error handling
"""
from fastapi import Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import time

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


def setup_cors(app, allowed_origins: list):
    """
    Setup CORS middleware

    Args:
        app: FastAPI application
        allowed_origins: List of allowed origins
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"]
    )
    logger.info(f"CORS configured for origins: {allowed_origins}")


def setup_rate_limiting(app):
    """
    Setup rate limiting middleware

    Args:
        app: FastAPI application
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("Rate limiting configured")


async def log_requests(request: Request, call_next):
    """
    Middleware to log all requests

    Args:
        request: Incoming request
        call_next: Next middleware in chain
    """
    start_time = time.time()

    # Log request
    logger.info(f"→ {request.method} {request.url.path}")

    try:
        response = await call_next(request)

        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"← {request.method} {request.url.path} "
            f"status={response.status_code} time={process_time:.3f}s"
        )

        # Add custom headers
        response.headers["X-Process-Time"] = str(process_time)

        return response

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error", "detail": str(e)}
        )


async def validate_content_type(request: Request, call_next):
    """
    Validate content type for POST/PUT requests

    Args:
        request: Incoming request
        call_next: Next middleware in chain
    """
    if request.method in ["POST", "PUT", "PATCH"]:
        content_type = request.headers.get("content-type", "")

        # Allow both JSON and multipart form data
        valid_types = ["application/json", "multipart/form-data"]

        if not any(vt in content_type for vt in valid_types):
            logger.warning(f"Invalid content type: {content_type}")
            return JSONResponse(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                content={
                    "error": "Unsupported Media Type",
                    "detail": "Content-Type must be application/json or multipart/form-data"
                }
            )

    return await call_next(request)


class SecurityHeadersMiddleware:
    """Add security headers to responses"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))

                # Add security headers
                headers[b"X-Content-Type-Options"] = b"nosniff"
                headers[b"X-Frame-Options"] = b"DENY"
                headers[b"X-XSS-Protection"] = b"1; mode=block"
                headers[b"Strict-Transport-Security"] = b"max-age=31536000; includeSubDomains"

                message["headers"] = list(headers.items())

            await send(message)

        await self.app(scope, receive, send_with_headers)
