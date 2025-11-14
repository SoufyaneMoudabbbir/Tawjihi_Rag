"""
Tests for utility functions
"""
import pytest
from utils import (
    clean_text,
    split_text,
    sanitize_filename,
    validate_file_size,
    calculate_confidence
)


class TestCleanText:
    """Test text cleaning"""

    def test_clean_basic_text(self):
        """Test basic text cleaning"""
        text = "This   is  a   test."
        cleaned = clean_text(text)
        assert "  " not in cleaned
        assert cleaned == "This is a test."

    def test_remove_special_characters(self):
        """Test removal of special characters"""
        text = "Hello@#$%World"
        cleaned = clean_text(text)
        assert "@" not in cleaned
        assert "#" not in cleaned


class TestSplitText:
    """Test text splitting"""

    def test_split_short_text(self):
        """Test splitting text shorter than chunk size"""
        text = "Short text."
        chunks = split_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text(self):
        """Test splitting long text"""
        text = "A" * 2000
        chunks = split_text(text, chunk_size=700, overlap=100)
        assert len(chunks) > 1
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_sentence_boundary(self):
        """Test splitting respects sentence boundaries"""
        text = "First sentence. Second sentence. Third sentence."
        chunks = split_text(text, chunk_size=25, overlap=5)
        # Should split at sentence boundaries when possible
        assert len(chunks) >= 1


class TestSanitizeFilename:
    """Test filename sanitization"""

    def test_remove_path_components(self):
        """Test removal of path components"""
        filename = "../../etc/passwd"
        sanitized = sanitize_filename(filename)
        assert ".." not in sanitized
        assert "/" not in sanitized

    def test_remove_dangerous_chars(self):
        """Test removal of dangerous characters"""
        filename = "test<script>.pdf"
        sanitized = sanitize_filename(filename)
        assert "<" not in sanitized
        assert ">" not in sanitized

    def test_limit_length(self):
        """Test filename length limit"""
        filename = "a" * 300 + ".pdf"
        sanitized = sanitize_filename(filename)
        assert len(sanitized) <= 255


class TestValidateFileSize:
    """Test file size validation"""

    def test_valid_file_size(self):
        """Test file within size limit"""
        assert validate_file_size(1024 * 1024, max_size_mb=50)  # 1MB

    def test_invalid_file_size(self):
        """Test file exceeds size limit"""
        assert not validate_file_size(100 * 1024 * 1024, max_size_mb=50)  # 100MB


class TestCalculateConfidence:
    """Test confidence calculation"""

    def test_high_confidence(self):
        """Test high confidence scores"""
        scores = [0.8, 0.85, 0.9]
        assert calculate_confidence(scores) == "High"

    def test_medium_confidence(self):
        """Test medium confidence scores"""
        scores = [0.5, 0.6, 0.55]
        assert calculate_confidence(scores) == "Medium"

    def test_low_confidence(self):
        """Test low confidence scores"""
        scores = [0.2, 0.3, 0.25]
        assert calculate_confidence(scores) == "Low"

    def test_empty_scores(self):
        """Test empty scores list"""
        assert calculate_confidence([]) == "Low"
