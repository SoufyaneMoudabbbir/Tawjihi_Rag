#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Validation Utilities
Validates file uploads for security
"""
import os
import magic
from pathlib import Path
from typing import BinaryIO, Optional
from app.core.config import settings
from app.core.exceptions import (
    InvalidFileError,
    FileTooLargeError,
    UnsupportedFileTypeError
)
from app.core.logging import get_logger

logger = get_logger(__name__)


class FileValidator:
    """Validates uploaded files"""

    # PDF magic numbers (file signatures)
    PDF_SIGNATURES = [
        b'%PDF-1.0',
        b'%PDF-1.1',
        b'%PDF-1.2',
        b'%PDF-1.3',
        b'%PDF-1.4',
        b'%PDF-1.5',
        b'%PDF-1.6',
        b'%PDF-1.7',
        b'%PDF-2.0',
    ]

    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """
        Validate file size
        Raises FileTooLargeError if too large
        """
        max_size = settings.max_file_size_bytes
        if file_size > max_size:
            raise FileTooLargeError(file_size, max_size)
        return True

    @staticmethod
    def validate_pdf_signature(file_data: bytes) -> bool:
        """
        Validate PDF file by checking magic numbers
        More secure than trusting MIME type
        """
        if len(file_data) < 8:
            raise InvalidFileError("File too small to be a valid PDF")

        # Check if file starts with PDF signature
        is_pdf = any(file_data.startswith(sig) for sig in FileValidator.PDF_SIGNATURES)

        if not is_pdf:
            raise InvalidFileError("File is not a valid PDF (invalid file signature)")

        return True

    @staticmethod
    def validate_filename(filename: str) -> str:
        """
        Sanitize and validate filename
        Prevents path traversal attacks
        """
        if not filename:
            raise InvalidFileError("Filename cannot be empty")

        # Remove path separators and null bytes
        filename = os.path.basename(filename)
        filename = filename.replace('\x00', '')
        filename = filename.replace('..', '')

        # Check for valid extension
        if not filename.lower().endswith('.pdf'):
            raise UnsupportedFileTypeError(
                Path(filename).suffix,
                ['.pdf']
            )

        # Limit filename length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext

        return filename

    @staticmethod
    def validate_pdf_structure(file_path: str) -> bool:
        """
        Validate PDF structure using PyPDF2
        Ensures file can be parsed
        """
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                # Try to read first page
                if len(pdf_reader.pages) > 0:
                    _ = pdf_reader.pages[0].extract_text()
                return True
        except Exception as e:
            logger.error(f"PDF structure validation failed: {e}")
            raise InvalidFileError(f"Invalid PDF structure: {str(e)}")


async def validate_pdf_file(
    file_data: bytes,
    filename: str,
    file_size: int
) -> str:
    """
    Complete PDF file validation pipeline

    Args:
        file_data: Raw file bytes
        filename: Original filename
        file_size: File size in bytes

    Returns:
        Sanitized filename

    Raises:
        InvalidFileError: If file is invalid
        FileTooLargeError: If file exceeds size limit
        UnsupportedFileTypeError: If file type not allowed
    """
    logger.info(f"Validating file: {filename} ({file_size} bytes)")

    # 1. Validate size
    FileValidator.validate_file_size(file_size)

    # 2. Sanitize filename
    clean_filename = FileValidator.validate_filename(filename)

    # 3. Validate PDF signature (magic numbers)
    FileValidator.validate_pdf_signature(file_data)

    logger.info(f"File validation passed: {clean_filename}")
    return clean_filename
