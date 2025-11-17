#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structured Logging Configuration
Provides JSON-formatted logging with rotation
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from app.core.config import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""

    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['environment'] = settings.ENVIRONMENT


def setup_logging():
    """Configure application logging"""

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    if settings.LOG_FORMAT == "json":
        console_formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    if settings.LOG_FILE:
        # Parse rotation size
        rotation_size = _parse_size(settings.LOG_ROTATION)

        file_handler = RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=rotation_size,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

        if settings.LOG_FORMAT == "json":
            file_formatter = CustomJsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s %(pathname)s %(lineno)d'
            )
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(pathname)s:%(lineno)d]'
            )

        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Suppress verbose libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logger


def _parse_size(size_str: str) -> int:
    """Parse size string like '10 MB' to bytes"""
    size_str = size_str.strip().upper()
    if 'MB' in size_str:
        return int(size_str.replace('MB', '').strip()) * 1024 * 1024
    elif 'KB' in size_str:
        return int(size_str.replace('KB', '').strip()) * 1024
    elif 'GB' in size_str:
        return int(size_str.replace('GB', '').strip()) * 1024 * 1024 * 1024
    else:
        return int(size_str)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)
