import { NextResponse } from "next/server"
import { openDb } from "@/lib/db"

export async function POST(request) {
  try {
    const { userId, config } = await request.json()

    if (!userId || !config) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 })
    }

    const db = await openDb()

    // Create form_configs table if it doesn't exist
    await db.exec(`
      CREATE TABLE IF NOT EXISTS form_configs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        title TEXT NOT NULL,
        config TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `)

    // Insert new form configuration
    await db.run("INSERT INTO form_configs (user_id, title, config) VALUES (?, ?, ?)", [
      userId,
      config.title,
      JSON.stringify(config),
    ])

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error("Error saving form config:", error)
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
    const configs = await db.all("SELECT * FROM form_configs WHERE user_id = ? ORDER BY created_at DESC", [userId])

    const formattedConfigs = configs.map((config) => ({
      ...config,
      config: JSON.parse(config.config),
    }))

    return NextResponse.json({ configs: formattedConfigs })
  } catch (error) {
    console.error("Error fetching form configs:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
