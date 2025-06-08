import { NextResponse } from "next/server"
import { openDb } from "@/lib/db"

export async function POST(request) {
  try {
    const { userId, responses } = await request.json()

    if (!userId || !responses) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 })
    }

    // Validate responses structure
    if (typeof responses !== "object") {
      return NextResponse.json({ error: "Invalid responses format" }, { status: 400 })
    }

    const db = await openDb()

    // Create table with index if it doesn't exist
    await db.exec(`
      CREATE TABLE IF NOT EXISTS user_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        responses TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `)

    // Create index on user_id for faster queries
    await db.exec(`
      CREATE INDEX IF NOT EXISTS idx_user_responses_user_id ON user_responses(user_id)
    `)

    // Check if user already has responses
    const existingResponse = await db.get("SELECT id FROM user_responses WHERE user_id = ?", [userId])

    if (existingResponse) {
      // Update existing responses
      await db.run("UPDATE user_responses SET responses = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?", [
        JSON.stringify(responses),
        userId,
      ])
    } else {
      // Insert new responses
      await db.run("INSERT INTO user_responses (user_id, responses) VALUES (?, ?)", [userId, JSON.stringify(responses)])
    }

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error("Error saving responses:", error)
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
    
    // Force table creation
    await db.exec(`
      CREATE TABLE IF NOT EXISTS user_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        responses TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `)

    // Create index
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_user_responses_user_id ON user_responses(user_id)`)

    const response = await db.get("SELECT responses, updated_at FROM user_responses WHERE user_id = ?", [userId])

    if (!response) {
      return NextResponse.json({ error: "No responses found" }, { status: 404 })
    }

    try {
      const parsedResponses = JSON.parse(response.responses)
      return NextResponse.json({
        responses: parsedResponses,
        lastUpdated: response.updated_at,
      })
    } catch (parseError) {
      console.error("Error parsing stored responses:", parseError)
      return NextResponse.json({ error: "Invalid stored data format" }, { status: 500 })
    }
  } catch (error) {
    console.error("Error fetching responses:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
