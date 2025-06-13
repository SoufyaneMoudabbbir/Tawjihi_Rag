import { NextResponse } from "next/server"
import { openDb } from "@/lib/db"

export async function GET(request, { params }) {
  try {
    const resolvedParams = await params
    const courseId = parseInt(resolvedParams.courseId)
    
    const db = await openDb()

    const chapters = await db.all(`
      SELECT * FROM course_chapters 
      WHERE course_id = ?
      ORDER BY chapter_number
    `, [courseId])

    return NextResponse.json({ 
      success: true,
      chapters: chapters || [],
      total_chapters: chapters.length
    })
  } catch (error) {
    console.error("Error fetching chapters:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}