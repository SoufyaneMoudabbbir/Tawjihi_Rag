import { NextResponse } from "next/server"
import { openDb } from "@/lib/db"

export async function POST(request) {
  try {
    const db = await openDb()
    
    console.log('🗄️ Creating quiz system database tables...')
    
    // ✅ Table 1: Store AI-generated quizzes
    await db.exec(`
      CREATE TABLE IF NOT EXISTS chapter_quizzes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chapter_id INTEGER NOT NULL,
        quiz_data TEXT NOT NULL,           -- JSON quiz structure from AI
        difficulty TEXT DEFAULT 'medium',
        question_count INTEGER DEFAULT 5,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (chapter_id) REFERENCES course_chapters (id) ON DELETE CASCADE
      )
    `)
    console.log('✅ Created chapter_quizzes table')

    // ✅ Table 2: Track every quiz attempt (supports unlimited retakes)
    await db.exec(`
      CREATE TABLE IF NOT EXISTS user_quiz_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        quiz_id INTEGER NOT NULL,
        chapter_id INTEGER NOT NULL,
        course_id INTEGER NOT NULL,
        user_answers TEXT NOT NULL,        -- JSON: [0,1,2,1,0] (selected option indexes)
        score INTEGER NOT NULL,            -- Score out of 100 (0-100)
        total_questions INTEGER NOT NULL,
        correct_answers INTEGER NOT NULL,
        time_taken INTEGER DEFAULT 0,     -- Seconds to complete quiz
        completed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (quiz_id) REFERENCES chapter_quizzes (id),
        FOREIGN KEY (chapter_id) REFERENCES course_chapters (id),
        FOREIGN KEY (course_id) REFERENCES courses (id)
      )
    `)
    console.log('✅ Created user_quiz_attempts table')

    // ✅ Table 3: Track best scores and chapter completion (your progress logic)
    await db.exec(`
      CREATE TABLE IF NOT EXISTS user_chapter_progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        course_id INTEGER NOT NULL,
        chapter_id INTEGER NOT NULL,
        status TEXT DEFAULT 'locked',      -- 'locked', 'unlocked', 'completed'
        best_quiz_score INTEGER DEFAULT 0, -- Best score achieved (0-100)
        quiz_attempts INTEGER DEFAULT 0,   -- Total number of attempts
        last_attempt_at DATETIME,
        completed_at DATETIME,             -- When they first scored 70%+
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, course_id, chapter_id),
        FOREIGN KEY (course_id) REFERENCES courses (id),
        FOREIGN KEY (chapter_id) REFERENCES course_chapters (id)
      )
    `)
    console.log('✅ Created user_chapter_progress table')

    // ✅ Create indexes for performance
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_chapter_quizzes_chapter_id ON chapter_quizzes(chapter_id)`)
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_user_quiz_attempts_user_id ON user_quiz_attempts(user_id)`)
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_user_quiz_attempts_chapter_id ON user_quiz_attempts(chapter_id)`)
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_user_chapter_progress_user_course ON user_chapter_progress(user_id, course_id)`)
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_user_chapter_progress_status ON user_chapter_progress(status)`)
    console.log('✅ Created performance indexes')

    // ✅ Initialize progress for existing users (if any chapters exist)
    const existingChapters = await db.all(`
      SELECT DISTINCT cc.course_id, cc.id as chapter_id, c.user_id
      FROM course_chapters cc
      JOIN courses c ON cc.course_id = c.id
    `)

    for (const chapter of existingChapters) {
      // Initialize chapter 1 as unlocked, others as locked
      const initialStatus = chapter.chapter_id === 1 ? 'unlocked' : 'locked'
      
      await db.run(`
        INSERT OR IGNORE INTO user_chapter_progress 
        (user_id, course_id, chapter_id, status)
        VALUES (?, ?, ?, ?)
      `, [chapter.user_id, chapter.course_id, chapter.chapter_id, initialStatus])
    }
    console.log(`✅ Initialized progress for ${existingChapters.length} existing chapters`)

    return NextResponse.json({ 
      success: true, 
      message: "Quiz system database created successfully",
      tables_created: [
        "chapter_quizzes",
        "user_quiz_attempts", 
        "user_chapter_progress"
      ],
      indexes_created: 5,
      progress_records_initialized: existingChapters.length
    })
    
  } catch (error) {
    console.error("❌ Quiz migration error:", error)
    return NextResponse.json({ 
      error: "Database migration failed",
      details: error.message 
    }, { status: 500 })
  }
}