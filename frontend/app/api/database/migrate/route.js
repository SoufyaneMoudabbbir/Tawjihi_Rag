import { NextResponse } from "next/server"
import { openDb } from "@/lib/db"

export async function POST(request) {
  try {
    const db = await openDb()
    
    // Create the new chapter-related tables
    await db.exec(`
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

    await db.exec(`
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

    await db.exec(`
      CREATE TABLE IF NOT EXISTS chapter_quizzes_11 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chapter_id INTEGER NOT NULL,
        quiz_data TEXT NOT NULL,
        passing_score INTEGER DEFAULT 70,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (chapter_id) REFERENCES course_chapters (id) ON DELETE CASCADE
      )
    `)

    // Create indexes
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_course_chapters_course_id ON course_chapters(course_id)`)
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_chapter_content_chapter_id ON chapter_content(chapter_id)`)
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_chapter_quizzes_chapter_id ON chapter_quizzes(chapter_id)`)

    return NextResponse.json({ success: true, message: "Database migrated successfully" })
  } catch (error) {
    console.error("Migration error:", error)
    return NextResponse.json({ error: "Migration failed" }, { status: 500 })
  }
}