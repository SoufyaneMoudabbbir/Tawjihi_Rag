import { NextResponse } from "next/server"
import { openDb } from "@/lib/db"

export async function POST(request) {
  try {
    const { userId, title, courseId } = await request.json()

    if (!userId) {
      return NextResponse.json({ error: "User ID required" }, { status: 400 })
    }

    const db = await openDb()

    // Update chat_sessions table to include course_id
    await db.exec(`
      CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        course_id INTEGER,
        title TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (course_id) REFERENCES courses (id) ON DELETE SET NULL
      )
    `)

    // Create index on user_id and course_id for faster queries
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id)`)
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_chat_sessions_course_id ON chat_sessions(course_id)`)

    // Create chat_messages table with indexes
    await db.exec(`
      CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        type TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
      )
    `)

    await db.exec(`CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id)`)

    // Insert new chat session
    const result = await db.run(
      "INSERT INTO chat_sessions (user_id, course_id, title) VALUES (?, ?, ?)", 
      [userId, courseId || null, title || "Educational Learning Session"]
    )

    // If this is a course-specific session, get course info for context
    let initialMessage = "Hello! I'm your educational assistant. I've reviewed your learning profile and I'm here to help you with your studies. What would you like to learn about today?"
    
    if (courseId) {
      const course = await db.get("SELECT name FROM courses WHERE id = ?", [courseId])
      if (course) {
        initialMessage = `Hello! I'm ready to help you learn ${course.name}. I have access to your course materials and can explain concepts, answer questions, create practice problems, and help you study effectively. What would you like to explore today?`
        
        // Update course chat count
        await db.run("UPDATE courses SET chat_count = chat_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE id = ?", [courseId])
      }
    }

    // Add initial bot message
    await db.run("INSERT INTO chat_messages (session_id, type, content) VALUES (?, ?, ?)", [
      result.lastID,
      "bot",
      initialMessage
    ])

    return NextResponse.json({ sessionId: result.lastID })
  } catch (error) {
    console.error("Error creating chat session:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}


export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get("userId")
    const courseId = searchParams.get("courseId")

    if (!userId) {
      return NextResponse.json({ error: "User ID required" }, { status: 400 })
    }

    const db = await openDb()

    let query = `
      SELECT 
        cs.*,
        c.name as course_name,
        COUNT(cm.id) as message_count
      FROM chat_sessions cs
      LEFT JOIN courses c ON cs.course_id = c.id
      LEFT JOIN chat_messages cm ON cs.id = cm.session_id
      WHERE cs.user_id = ?
    `
    const params = [userId]

    if (courseId) {
      query += " AND cs.course_id = ?"
      params.push(courseId)
    }

    query += `
      GROUP BY cs.id
      ORDER BY cs.updated_at DESC
    `

    const sessions = await db.all(query, params)

    // Format sessions with course info
    const formattedSessions = sessions.map(session => ({
      ...session,
      courseName: session.course_name || 'General Learning',
      messageCount: session.message_count || 0
    }))

    return NextResponse.json({ sessions: formattedSessions })
  } catch (error) {
    console.error("Error fetching chat sessions:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
