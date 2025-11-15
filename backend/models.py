"""
Pydantic models for request/response validation
"""
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, validator


class QuestionRequest(BaseModel):
    """Request model for chat questions"""
    question: str = Field(..., min_length=1, max_length=5000, description="User's question")
    course_id: Optional[int] = Field(None, ge=1, description="Course ID for context")
    chapter_id: Optional[int] = Field(None, ge=1, description="Chapter ID for specific chapter chat")
    user_id: str = Field(..., min_length=1, max_length=255, description="User identifier")
    user_profile: Optional[Dict] = Field(None, description="User learning profile")
    stream: bool = Field(True, description="Enable streaming response")

    @validator('question')
    def validate_question(cls, v):
        """Validate and sanitize question"""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        # Basic sanitization
        v = v.strip()
        if len(v) > 5000:
            raise ValueError("Question too long (max 5000 characters)")
        return v


class ChatResponse(BaseModel):
    """Response model for chat answers"""
    response: str
    sources_count: int = Field(ge=0)
    confidence: str = Field(pattern="^(High|Medium|Low)$")
    avg_score: float = Field(ge=0.0, le=1.0)
    top_score: float = Field(ge=0.0, le=1.0)
    course_name: Optional[str] = None
    chapter_name: Optional[str] = None
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    courses_loaded: int = Field(ge=0)
    total_documents: int = Field(ge=0)
    version: str = "1.0.0"


class CourseCreateRequest(BaseModel):
    """Request to create a new course"""
    user_id: str = Field(..., min_length=1, max_length=255)
    name: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=2000)
    professor: Optional[str] = Field(None, max_length=255)
    semester: Optional[str] = Field(None, max_length=100)


class ChapterAnalysisRequest(BaseModel):
    """Request for chapter analysis"""
    course_id: int = Field(..., ge=1)
    user_id: str = Field(..., min_length=1, max_length=255)


class QuizSubmissionRequest(BaseModel):
    """Quiz submission"""
    user_id: str = Field(..., min_length=1, max_length=255)
    chapter_id: int = Field(..., ge=1)
    answers: Dict[str, str] = Field(..., description="Question ID to answer mapping")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
