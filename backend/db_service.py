"""
Database service layer for SQLite operations
Handles all database interactions with proper connection pooling
"""
import sqlite3
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)


class DatabaseService:
    """Thread-safe database service with connection pooling"""

    def __init__(self, database_path: str):
        self.database_path = database_path
        self._local = threading.local()

        # Ensure database file exists
        db_path = Path(database_path)
        if not db_path.parent.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Database initialized at: {database_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.database_path,
                check_same_thread=True,  # Fixed: proper threading
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection

    @contextmanager
    def get_db(self):
        """Context manager for database operations"""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            pass  # Don't close, keep connection for thread

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a query with parameters"""
        with self.get_db() as conn:
            return conn.execute(query, params)

    def fetchone(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """Fetch one row"""
        with self.get_db() as conn:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None

    def fetchall(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Fetch all rows"""
        with self.get_db() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_course_info(self, course_id: int) -> Optional[Dict[str, Any]]:
        """Get course information"""
        return self.fetchone(
            "SELECT * FROM courses WHERE id = ?",
            (course_id,)
        )

    def get_course_files(self, course_id: int) -> List[Dict[str, Any]]:
        """Get all files for a course"""
        return self.fetchall(
            """SELECT id, course_id, filename, original_name, file_path, file_size,
                      upload_date, processed
               FROM course_files
               WHERE course_id = ?
               ORDER BY upload_date DESC""",
            (course_id,)
        )

    def get_chapter_content(self, chapter_id: int) -> List[Dict[str, Any]]:
        """Get chapter content"""
        return self.fetchall(
            """SELECT * FROM chapter_content
               WHERE chapter_id = ?
               ORDER BY vector_index""",
            (chapter_id,)
        )

    def save_chapter_structure(
        self,
        course_id: int,
        chapter_data: Dict[str, Any]
    ) -> int:
        """Save chapter structure to database"""
        with self.get_db() as conn:
            cursor = conn.execute(
                """INSERT INTO course_chapters
                   (course_id, chapter_number, title, content_summary,
                    difficulty_level, estimated_study_time, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    course_id,
                    chapter_data.get('chapter_number'),
                    chapter_data.get('title'),
                    chapter_data.get('summary'),
                    chapter_data.get('difficulty', 'medium'),
                    chapter_data.get('study_time', 30),
                    'unlocked'
                )
            )
            return cursor.lastrowid

    def track_user_progress(
        self,
        user_id: str,
        course_id: int,
        activity_type: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Track user learning progress"""
        import json
        with self.get_db() as conn:
            conn.execute(
                """INSERT INTO user_progress
                   (user_id, course_id, activity_type, metadata, created_at)
                   VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (user_id, course_id, activity_type, json.dumps(metadata or {}))
            )

    def close(self):
        """Close database connection"""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
