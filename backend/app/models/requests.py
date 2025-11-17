#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Request Models
Pydantic models for API request validation
"""
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, validator


class QuestionRequest(BaseModel):
    """Chat question request"""
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    course_id: Optional[int] = Field(None, ge=1, description="Course ID for context")
    user_id: str = Field(..., min_length=1, max_length=255, description="User identifier")
    user_profile: Optional[Dict] = Field(None, description="User learning profile")
    stream: bool = Field(default=True, description="Enable streaming response")

    @validator("question")
    def validate_question(cls, v):
        """Validate and sanitize question"""
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        # Basic sanitization - remove potentially harmful characters
        dangerous_chars = ['<', '>', '{', '}', '\\x00']
        for char in dangerous_chars:
            if char in v:
                v = v.replace(char, '')
        return v


class ChapterChatRequest(BaseModel):
    """Chapter-specific chat request"""
    question: str = Field(..., min_length=1, max_length=2000)
    course_id: int = Field(..., ge=1)
    chapter_id: int = Field(..., ge=1)
    user_id: str = Field(..., min_length=1, max_length=255)
    user_profile: Optional[Dict] = None

    @validator("question")
    def validate_question(cls, v):
        """Validate and sanitize question"""
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        return v


class AnalyzeCourseRequest(BaseModel):
    """Course structure analysis request"""
    course_id: int = Field(..., ge=1, description="Course ID to analyze")
    user_id: str = Field(..., min_length=1, max_length=255)
    force_reanalysis: bool = Field(default=False, description="Force re-analysis even if exists")

    @validator("course_id")
    def validate_course_id(cls, v):
        """Validate course ID"""
        if v < 1:
            raise ValueError("Invalid course ID")
        return v


class QuizSubmissionRequest(BaseModel):
    """Quiz answer submission"""
    user_id: str = Field(..., min_length=1, max_length=255)
    quiz_id: int = Field(..., ge=1)
    chapter_id: int = Field(..., ge=1)
    course_id: int = Field(..., ge=1)
    user_answers: List[int] = Field(..., min_items=1, max_items=50)
    time_taken: int = Field(default=0, ge=0, description="Time taken in seconds")

    @validator("user_answers")
    def validate_answers(cls, v):
        """Validate answer indices"""
        if not v:
            raise ValueError("No answers provided")
        # Ensure all answers are valid option indices (0-3 typically)
        for answer in v:
            if answer < 0 or answer > 10:
                raise ValueError(f"Invalid answer index: {answer}")
        return v


class FileUploadMetadata(BaseModel):
    """File upload metadata validation"""
    filename: str = Field(..., min_length=1, max_length=255)
    file_size: int = Field(..., ge=0)
    content_type: str = Field(..., min_length=1)

    @validator("filename")
    def validate_filename(cls, v):
        """Validate filename"""
        # Remove path traversal attempts
        v = v.replace('..', '').replace('/', '').replace('\\', '')
        # Remove null bytes
        v = v.replace('\x00', '')
        if not v:
            raise ValueError("Invalid filename")
        return v

    @validator("content_type")
    def validate_content_type(cls, v):
        """Validate content type"""
        allowed_types = ['application/pdf']
        if v not in allowed_types:
            raise ValueError(f"Content type must be one of: {allowed_types}")
        return v
