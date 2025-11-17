#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Connection Manager
Thread-safe async SQLite connection manager
"""
import sqlite3
import aiosqlite
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
import threading
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """
    Thread-safe database connection manager
    Uses aiosqlite for async operations
    """

    def __init__(self, database_path: str):
        self.database_path = database_path
        self._local = threading.local()
        self._connection: Optional[aiosqlite.Connection] = None
        self._lock = threading.Lock()

    async def connect(self) -> aiosqlite.Connection:
        """Create async database connection"""
        if self._connection is None:
            self._connection = await aiosqlite.connect(
                self.database_path,
                check_same_thread=True,  # ✅ FIXED: Enable thread safety
                timeout=10.0
            )
            # Enable foreign keys
            await self._connection.execute("PRAGMA foreign_keys = ON")
            # Set journal mode to WAL for better concurrency
            await self._connection.execute("PRAGMA journal_mode = WAL")
            # Row factory for dict-like access
            self._connection.row_factory = aiosqlite.Row
            logger.info(f"Database connected: {self.database_path}")
        return self._connection

    async def disconnect(self):
        """Close database connection"""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database disconnected")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """
        Get database connection as context manager
        Ensures proper cleanup
        """
        conn = await self.connect()
        try:
            yield conn
        finally:
            # Don't close the connection, keep it alive
            # It will be closed on app shutdown
            pass

    async def execute(self, query: str, parameters: tuple = ()):
        """Execute a query"""
        async with self.get_connection() as conn:
            cursor = await conn.execute(query, parameters)
            await conn.commit()
            return cursor

    async def fetchone(self, query: str, parameters: tuple = ()):
        """Fetch one row"""
        async with self.get_connection() as conn:
            cursor = await conn.execute(query, parameters)
            return await cursor.fetchone()

    async def fetchall(self, query: str, parameters: tuple = ()):
        """Fetch all rows"""
        async with self.get_connection() as conn:
            cursor = await conn.execute(query, parameters)
            return await cursor.fetchall()

    async def execute_many(self, query: str, parameters: list):
        """Execute query with multiple parameter sets"""
        async with self.get_connection() as conn:
            await conn.executemany(query, parameters)
            await conn.commit()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get or create database manager singleton"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(settings.DATABASE_PATH)
    return _db_manager


async def get_db() -> AsyncGenerator[DatabaseManager, None]:
    """
    Dependency for FastAPI endpoints
    Usage: db: DatabaseManager = Depends(get_db)
    """
    db_manager = get_database_manager()
    yield db_manager


async def init_database():
    """Initialize database on startup"""
    db_manager = get_database_manager()
    await db_manager.connect()
    logger.info("Database initialized")


async def close_database():
    """Close database on shutdown"""
    global _db_manager
    if _db_manager:
        await _db_manager.disconnect()
        _db_manager = None
    logger.info("Database closed")
