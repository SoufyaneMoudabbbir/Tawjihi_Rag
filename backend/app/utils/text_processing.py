#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Processing Utilities
Handles text cleaning and chunking
"""
import re
from typing import List
from app.core.config import settings


class TextProcessor:
    """Text processing utilities for RAG system"""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text for educational content

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

    @staticmethod
    def split_text(
        text: str,
        chunk_size: int = None,
        overlap: int = None
    ) -> List[str]:
        """
        Split text into chunks with overlap
        Tries to split at sentence boundaries

        Args:
            text: Text to split
            chunk_size: Maximum chunk size (default from config)
            overlap: Overlap between chunks (default from config)

        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP

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

            # Only add chunks with meaningful content
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())

            start = end - overlap

        return chunks

    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from text (simple implementation)

        Args:
            text: Text to extract keywords from
            top_n: Number of keywords to return

        Returns:
            List of keywords
        """
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }

        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter and count
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        return [word for word, _ in sorted_words[:top_n]]
