"""
Utility functions for text processing and validation
"""
import re
from typing import List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Enhanced text cleaning for educational content

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text
    """
    # Remove non-useful characters but preserve educational formatting
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\(\)\-\+\=\%\$\#\@\&\*\/\\\[\]\{\}\|]', '', text)

    # Clean up spacing and formatting
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    text = re.sub(r'\s*([,;:])\s*', r'\1 ', text)

    # Remove excessive punctuation
    text = re.sub(r'[.\-_]{3,}', ' ', text)
    text = re.sub(r'=+', ' ', text)

    return text.strip()


def split_text(
    text: str,
    chunk_size: int = 700,
    overlap: int = 100
) -> List[str]:
    """
    Improved text splitting for educational content with sentence boundary awareness

    Args:
        text: Text to split
        chunk_size: Target size for each chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to end at sentence boundary
        if end < len(text):
            # Look for sentence endings
            sentence_endings = [chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?')]
            best_cut = max([pos for pos in sentence_endings if pos > start + chunk_size // 2] + [-1])

            if best_cut > -1:
                chunk = text[start:start + best_cut + 1]
                end = start + best_cut + 1
            else:
                # Fallback to paragraph break
                last_paragraph = chunk.rfind('\n\n')
                if last_paragraph > start + chunk_size // 2:
                    chunk = text[start:start + last_paragraph]
                    end = start + last_paragraph

        if len(chunk.strip()) > 50:
            chunks.append(chunk.strip())

        start = end - overlap

    return chunks


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove any path components
    filename = Path(filename).name

    # Remove potentially dangerous characters
    filename = re.sub(r'[^\w\s\-\.]', '_', filename)

    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:250] + ('.' + ext if ext else '')

    return filename


def validate_file_size(file_size: int, max_size_mb: int = 50) -> bool:
    """
    Validate file size

    Args:
        file_size: File size in bytes
        max_size_mb: Maximum allowed size in MB

    Returns:
        True if valid, False otherwise
    """
    max_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_bytes


def calculate_confidence(scores: List[float]) -> str:
    """
    Calculate confidence level based on similarity scores

    Args:
        scores: List of similarity scores

    Returns:
        Confidence level: "High", "Medium", or "Low"
    """
    if not scores:
        return "Low"

    avg_score = sum(scores) / len(scores)

    if avg_score > 0.7:
        return "High"
    elif avg_score > 0.4:
        return "Medium"
    else:
        return "Low"


def format_timestamp() -> str:
    """Get current timestamp in ISO format"""
    from datetime import datetime
    return datetime.utcnow().isoformat()


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract keywords from text

    Args:
        text: Text to extract keywords from
        min_length: Minimum keyword length

    Returns:
        List of keywords
    """
    # Basic keyword extraction (can be enhanced with NLP)
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [w for w in words if len(w) >= min_length]

    # Remove common stop words (simplified)
    stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'in', 'with', 'to', 'for'}
    keywords = [w for w in keywords if w not in stop_words]

    return list(set(keywords))
