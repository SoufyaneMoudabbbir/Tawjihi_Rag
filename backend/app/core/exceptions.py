#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Exception Classes
Provides structured error handling
"""
from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class TawjihiBaseException(Exception):
    """Base exception for all custom exceptions"""

    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class DatabaseError(TawjihiBaseException):
    """Database operation errors"""

    def __init__(self, message: str = "Database operation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR, details)


class CourseNotFoundError(TawjihiBaseException):
    """Course not found"""

    def __init__(self, course_id: int):
        super().__init__(
            f"Course with ID {course_id} not found",
            status.HTTP_404_NOT_FOUND,
            {"course_id": course_id}
        )


class ChapterNotFoundError(TawjihiBaseException):
    """Chapter not found"""

    def __init__(self, chapter_id: int):
        super().__init__(
            f"Chapter with ID {chapter_id} not found",
            status.HTTP_404_NOT_FOUND,
            {"chapter_id": chapter_id}
        )


class InvalidFileError(TawjihiBaseException):
    """Invalid file upload"""

    def __init__(self, message: str = "Invalid file", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status.HTTP_400_BAD_REQUEST, details)


class FileTooLargeError(TawjihiBaseException):
    """File exceeds size limit"""

    def __init__(self, file_size: int, max_size: int):
        super().__init__(
            f"File size {file_size} bytes exceeds maximum {max_size} bytes",
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            {"file_size": file_size, "max_size": max_size}
        )


class UnsupportedFileTypeError(TawjihiBaseException):
    """Unsupported file type"""

    def __init__(self, file_type: str, allowed_types: list):
        super().__init__(
            f"File type '{file_type}' not supported. Allowed: {', '.join(allowed_types)}",
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            {"file_type": file_type, "allowed_types": allowed_types}
        )


class ValidationError(TawjihiBaseException):
    """Input validation error"""

    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(message, status.HTTP_422_UNPROCESSABLE_ENTITY, details)


class RAGServiceError(TawjihiBaseException):
    """RAG service errors"""

    def __init__(self, message: str = "RAG service error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR, details)


class EmbeddingError(TawjihiBaseException):
    """Embedding generation errors"""

    def __init__(self, message: str = "Failed to generate embeddings"):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)


class DeepSeekAPIError(TawjihiBaseException):
    """DeepSeek API errors"""

    def __init__(self, message: str, status_code: int = 500, api_response: Optional[str] = None):
        details = {"api_response": api_response} if api_response else {}
        super().__init__(message, status_code, details)


class RateLimitExceededError(TawjihiBaseException):
    """Rate limit exceeded"""

    def __init__(self, retry_after: Optional[int] = None):
        message = "Rate limit exceeded"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, status.HTTP_429_TOO_MANY_REQUESTS, {"retry_after": retry_after})


class AuthenticationError(TawjihiBaseException):
    """Authentication errors"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status.HTTP_401_UNAUTHORIZED)


class AuthorizationError(TawjihiBaseException):
    """Authorization errors"""

    def __init__(self, message: str = "Access denied"):
        super().__init__(message, status.HTTP_403_FORBIDDEN)


class ServiceUnavailableError(TawjihiBaseException):
    """Service temporarily unavailable"""

    def __init__(self, service_name: str = "Service"):
        super().__init__(
            f"{service_name} temporarily unavailable",
            status.HTTP_503_SERVICE_UNAVAILABLE,
            {"service": service_name}
        )
