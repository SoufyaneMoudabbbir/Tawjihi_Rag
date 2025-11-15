"""
Tests for Pydantic models
"""
import pytest
from pydantic import ValidationError
from models import QuestionRequest, ChatResponse, HealthResponse


class TestQuestionRequest:
    """Test QuestionRequest model"""

    def test_valid_question_request(self):
        """Test valid question request"""
        data = {
            "question": "What is photosynthesis?",
            "course_id": 1,
            "user_id": "user123",
            "stream": True
        }
        request = QuestionRequest(**data)
        assert request.question == "What is photosynthesis?"
        assert request.course_id == 1
        assert request.user_id == "user123"

    def test_question_too_long(self):
        """Test question exceeds max length"""
        data = {
            "question": "A" * 6000,  # Exceeds 5000 limit
            "user_id": "user123"
        }
        with pytest.raises(ValidationError):
            QuestionRequest(**data)

    def test_empty_question(self):
        """Test empty question"""
        data = {
            "question": "   ",
            "user_id": "user123"
        }
        with pytest.raises(ValidationError):
            QuestionRequest(**data)

    def test_missing_user_id(self):
        """Test missing required user_id"""
        data = {
            "question": "Test question"
        }
        with pytest.raises(ValidationError):
            QuestionRequest(**data)


class TestChatResponse:
    """Test ChatResponse model"""

    def test_valid_chat_response(self):
        """Test valid chat response"""
        data = {
            "response": "Photosynthesis is...",
            "sources_count": 3,
            "confidence": "High",
            "avg_score": 0.85,
            "top_score": 0.92,
            "timestamp": "2024-01-01T00:00:00"
        }
        response = ChatResponse(**data)
        assert response.confidence == "High"
        assert response.sources_count == 3

    def test_invalid_confidence(self):
        """Test invalid confidence value"""
        data = {
            "response": "Test",
            "sources_count": 1,
            "confidence": "Invalid",  # Must be High/Medium/Low
            "avg_score": 0.5,
            "top_score": 0.5,
            "timestamp": "2024-01-01T00:00:00"
        }
        with pytest.raises(ValidationError):
            ChatResponse(**data)


class TestHealthResponse:
    """Test HealthResponse model"""

    def test_valid_health_response(self):
        """Test valid health response"""
        data = {
            "status": "healthy",
            "model_loaded": True,
            "courses_loaded": 5,
            "total_documents": 100
        }
        response = HealthResponse(**data)
        assert response.status == "healthy"
        assert response.courses_loaded == 5
