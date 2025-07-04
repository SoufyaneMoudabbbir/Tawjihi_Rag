import { NextResponse } from "next/server"
import { openDb } from "@/lib/db"

export async function GET(request, { params }) {
  try {
    const { sessionId } = await params
    const db = await openDb()
    

    // Get session details
    const session = await db.get("SELECT * FROM chat_sessions WHERE id = ?", [sessionId])

    if (!session) {
      return NextResponse.json({ error: "Session not found" }, { status: 404 })
    }

    // Get all messages for this session
    const messages = await db.all(
      `
      SELECT * FROM chat_messages 
      WHERE session_id = ? 
      ORDER BY timestamp ASC
    `,
      [sessionId],
    )

    // Format messages for frontend
    const formattedMessages = messages.map((msg) => ({
      id: msg.id,
      type: msg.type,
      content: msg.content,
      timestamp: new Date(msg.timestamp),
    }))

    let course = null
      if (session.course_id) {
        course = await db.get("SELECT id, name, professor, semester FROM courses WHERE id = ?", [session.course_id])
      }

      let sessionMetadata = {}
        try {
          sessionMetadata = session.metadata ? JSON.parse(session.metadata) : {}
        } catch (e) {
          console.error('Error parsing session metadata:', e)
          sessionMetadata = {}
        }

        return NextResponse.json({
          title: session.title,
          messages: formattedMessages,
          course: course,
          session: {
            id: session.id,
            course_id: session.course_id,
            session_type: session.session_type,  // ✅ Add session type
            metadata: sessionMetadata,           // ✅ Add parsed metadata (chapter info)
            created_at: session.created_at,
            updated_at: session.updated_at
          }
        })
  } catch (error) {
    console.error("Error fetching chat session:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
export async function PATCH(request, { params }) {
  try {
    const sessionId = params.sessionId
    const { title } = await request.json()

    if (!title) {
      return NextResponse.json({ error: "Title is required" }, { status: 400 })
    }

    const db = await openDb()
    
    await db.run(
      "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
      [title, sessionId]
    )

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error("Error updating session title:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}


export async function DELETE(request, { params }) {
  try {
    const sessionId = params.sessionId
    const db = await openDb()

    // Delete messages first (foreign key constraint)
    await db.run("DELETE FROM chat_messages WHERE session_id = ?", [sessionId])

    // Delete session
    await db.run("DELETE FROM chat_sessions WHERE id = ?", [sessionId])

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error("Error deleting chat session:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
