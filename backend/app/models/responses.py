#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Response Models
Pydantic models for API responses
"""
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="AI response text")
    sources_count: int = Field(..., ge=0, description="Number of source documents used")
    confidence: str = Field(..., description="Confidence level: High, Medium, Low")
    avg_score: float = Field(..., ge=0.0, le=1.0, description="Average relevance score")
    top_score: float = Field(..., ge=0.0, le=1.0, description="Top relevance score")
    course_name: Optional[str] = Field(None, description="Course name if applicable")
    timestamp: str = Field(..., description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether embedding model is loaded")
    courses_loaded: int = Field(..., ge=0, description="Number of courses loaded")
    total_documents: int = Field(..., ge=0, description="Total document chunks loaded")
    version: str = Field(default="2.0.0", description="API version")
    environment: str = Field(..., description="Environment: development/staging/production")


class CourseLoadResponse(BaseModel):
    """Course loading response"""
    success: bool
    course_id: int
    course_name: str
    documents_loaded: int = Field(..., ge=0)
    message: str


class QuizQuestionResponse(BaseModel):
    """Single quiz question"""
    question: str
    options: List[str] = Field(..., min_items=2, max_items=6)
    correct: int = Field(..., ge=0)
    explanation: Optional[str] = None


class QuizMetadata(BaseModel):
    """Quiz metadata"""
    difficulty: str = Field(default="medium")
    estimated_time: int = Field(..., ge=0, description="Estimated time in minutes")
    passing_score: int = Field(default=70, ge=0, le=100)
    total_questions: int = Field(..., ge=1)


class QuizResponse(BaseModel):
    """Quiz data response"""
    success: bool
    quiz_id: Optional[int] = None
    chapter_id: int
    course_id: Optional[int] = None
    questions: List[QuizQuestionResponse]
    quiz_metadata: QuizMetadata
    source: str = Field(default="generated", description="existing or generated")


class QuizSubmissionResponse(BaseModel):
    """Quiz submission result"""
    success: bool
    score: int = Field(..., ge=0, le=100)
    correct_answers: int = Field(..., ge=0)
    total_questions: int = Field(..., ge=1)
    passed: bool
    previous_best: int = Field(..., ge=0, le=100)
    new_best: int = Field(..., ge=0, le=100)
    status: str = Field(..., description="unlocked or completed")


class ChapterProgressResponse(BaseModel):
    """Chapter progress data"""
    chapter_id: int
    status: str = Field(..., description="locked, unlocked, or completed")
    best_score: Optional[int] = Field(None, ge=0, le=100)
    attempts: int = Field(default=0, ge=0)
    title: str
    chapter_number: int


class UserProgressResponse(BaseModel):
    """User progress for a course"""
    success: bool
    user_id: str
    course_id: int
    progress_records: List[ChapterProgressResponse]
    total_chapters: int = Field(..., ge=0)
    completed_chapters: int = Field(..., ge=0)


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    path: Optional[str] = Field(None, description="Request path that caused the error")


class LoadedCoursesResponse(BaseModel):
    """Response for loaded courses endpoint"""
    courses: List[Dict[str, Any]]


class ExamplesResponse(BaseModel):
    """Example questions response"""
    examples: List[str] = Field(..., min_items=1)


class AnalyzeCourseResponse(BaseModel):
    """Course analysis response"""
    success: bool
    course_id: int
    chapters_found: int = Field(..., ge=0)
    chapters: List[Dict[str, Any]]
    message: str
