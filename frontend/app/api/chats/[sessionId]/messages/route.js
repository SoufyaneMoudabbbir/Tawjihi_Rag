import { NextResponse } from "next/server"
import { openDb } from "@/lib/db"

export async function POST(request, { params }) {
  try {
    const { sessionId } = await params

    const { message } = await request.json()

    if (!message || !message.content) {
      return NextResponse.json({ error: "Message content required" }, { status: 400 })
    }

    const db = await openDb()

    // Insert message
    await db.run("INSERT INTO chat_messages (session_id, type, content) VALUES (?, ?, ?)", [
      sessionId,
      message.type,
      message.content,
    ])

    // Update session timestamp
    await db.run("UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", [sessionId])

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error("Error saving message:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
