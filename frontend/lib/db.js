import sqlite3 from "sqlite3"
import { open } from "sqlite"
import path from "path"

let db = null
let initialized = false

async function initializeDatabase(database) {
  if (initialized) return

  console.log("🔧 Initializing database schema...")

  // ==============================
  // CORE TABLES
  // ==============================

  // 1. Courses table
  await database.exec(`
    CREATE TABLE IF NOT EXISTS courses (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT NOT NULL,
      name TEXT NOT NULL,
      description TEXT,
      professor TEXT,
      semester TEXT,
      status TEXT DEFAULT 'active',
      file_count INTEGER DEFAULT 0,
      chat_count INTEGER DEFAULT 0,
      progress INTEGER DEFAULT 0,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      last_accessed DATETIME
    )
  `)

  // 2. Course files table
  await database.exec(`
    CREATE TABLE IF NOT EXISTS course_files (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      course_id INTEGER NOT NULL,
      filename TEXT NOT NULL,
      original_name TEXT NOT NULL,
      file_path TEXT NOT NULL,
      file_size INTEGER,
      upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
      processed BOOLEAN DEFAULT FALSE,
      FOREIGN KEY (course_id) REFERENCES courses (id) ON DELETE CASCADE
    )
  `)

  // 3. Chat sessions table (with session_type column)
  await database.exec(`
    CREATE TABLE IF NOT EXISTS chat_sessions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT NOT NULL,
      course_id INTEGER,
      title TEXT NOT NULL,
      session_type TEXT DEFAULT 'general',
      metadata TEXT DEFAULT '{}',
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (course_id) REFERENCES courses (id) ON DELETE SET NULL
    )
  `)

  // 4. Chat messages table
  await database.exec(`
    CREATE TABLE IF NOT EXISTS chat_messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id INTEGER NOT NULL,
      type TEXT NOT NULL,
      content TEXT NOT NULL,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
    )
  `)

  // 5. User responses table (for questionnaires)
  await database.exec(`
    CREATE TABLE IF NOT EXISTS user_responses (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT NOT NULL,
      responses TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `)

  // ==============================
  // CHAPTER-RELATED TABLES
  // ==============================

  // 6. Course chapters table
  await database.exec(`
    CREATE TABLE IF NOT EXISTS course_chapters (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      course_id INTEGER NOT NULL,
      chapter_number INTEGER NOT NULL,
      title TEXT NOT NULL,
      content_summary TEXT,
      estimated_study_time INTEGER DEFAULT 30,
      difficulty_level TEXT DEFAULT 'medium',
      prerequisites TEXT,
      status TEXT DEFAULT 'locked',
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (course_id) REFERENCES courses (id) ON DELETE CASCADE
    )
  `)

  // 7. Chapter content table
  await database.exec(`
    CREATE TABLE IF NOT EXISTS chapter_content (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      chapter_id INTEGER NOT NULL,
      content_type TEXT NOT NULL,
      content_text TEXT,
      page_reference TEXT,
      vector_index INTEGER,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (chapter_id) REFERENCES course_chapters (id) ON DELETE CASCADE
    )
  `)

  // 8. Chapter quizzes table
  await database.exec(`
    CREATE TABLE IF NOT EXISTS chapter_quizzes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      chapter_id INTEGER NOT NULL,
      quiz_data TEXT NOT NULL,
      passing_score INTEGER DEFAULT 70,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (chapter_id) REFERENCES course_chapters (id) ON DELETE CASCADE
    )
  `)

  // ==============================
  // INDEXES FOR PERFORMANCE
  // ==============================

  // Courses indexes
  await database.exec(`CREATE INDEX IF NOT EXISTS idx_courses_user_id ON courses(user_id)`)

  // Course files indexes
  await database.exec(`CREATE INDEX IF NOT EXISTS idx_course_files_course_id ON course_files(course_id)`)

  // Chat sessions indexes
  await database.exec(`CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id)`)
  await database.exec(`CREATE INDEX IF NOT EXISTS idx_chat_sessions_course_id ON chat_sessions(course_id)`)
  await database.exec(`CREATE INDEX IF NOT EXISTS idx_chat_sessions_type ON chat_sessions(session_type)`)

  // Chat messages indexes
  await database.exec(`CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id)`)

  // User responses indexes
  await database.exec(`CREATE INDEX IF NOT EXISTS idx_user_responses_user_id ON user_responses(user_id)`)

  // Chapter indexes
  await database.exec(`CREATE INDEX IF NOT EXISTS idx_course_chapters_course_id ON course_chapters(course_id)`)
  await database.exec(`CREATE INDEX IF NOT EXISTS idx_chapter_content_chapter_id ON chapter_content(chapter_id)`)
  await database.exec(`CREATE INDEX IF NOT EXISTS idx_chapter_quizzes_chapter_id ON chapter_quizzes(chapter_id)`)

  initialized = true
  console.log("✅ Database schema initialized successfully")
}

export async function openDb() {
  if (!db) {
    db = await open({
      filename: path.join(process.cwd(), "database.sqlite"),
      driver: sqlite3.Database,
    })

    // Initialize database schema on first connection
    await initializeDatabase(db)
  }
  return db
}
