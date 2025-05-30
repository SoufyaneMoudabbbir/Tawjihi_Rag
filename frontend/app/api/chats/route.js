import { NextResponse } from "next/server"
import { openDb } from "@/lib/db"

export async function POST(request) {
  try {
    const { userId, title } = await request.json()

    if (!userId) {
      return NextResponse.json({ error: "User ID required" }, { status: 400 })
    }

    const db = await openDb()

    // Create chat_sessions table with indexes
    await db.exec(`
      CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        title TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `)

    // Create index on user_id for faster queries
    await db.exec(`
      CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id)
    `)

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

    // Create index on session_id for faster queries
    await db.exec(`
      CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id)
    `)

    // Insert new chat session
    const result = await db.run("INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)", [
      userId,
      title || "Educational Path Exploration",
    ])

    // Add initial bot message
    await db.run("INSERT INTO chat_messages (session_id, type, content) VALUES (?, ?, ?)", [
      result.lastID,
      "bot",
      "Hello! I'm your educational guidance assistant. I've reviewed your questionnaire responses and I'm here to help you explore educational opportunities in Morocco. What would you like to know?",
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

    if (!userId) {
      return NextResponse.json({ error: "User ID required" }, { status: 400 })
    }

    const db = await openDb()

    // Get all chat sessions for user with message count
    const sessions = await db.all(
      `
      SELECT 
        cs.*,
        COUNT(cm.id) as message_count
      FROM chat_sessions cs
      LEFT JOIN chat_messages cm ON cs.id = cm.session_id
      WHERE cs.user_id = ?
      GROUP BY cs.id
      ORDER BY cs.updated_at DESC
    `,
      [userId],
    )

    return NextResponse.json({ sessions })
  } catch (error) {
    console.error("Error fetching chat sessions:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
