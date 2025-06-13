import { NextResponse } from "next/server"
import { openDb } from "@/lib/db"

export async function GET(request, { params }) {
  try {
    // Fix: Await params in Next.js 15
    const { courseId } = await params
    const db = await openDb()

    // Get chapters for this course
    const chapters = await db.all(`
      SELECT 
        cc.*,
        COUNT(cq.id) as quiz_count
      FROM course_chapters cc
      LEFT JOIN chapter_quizzes cq ON cc.id = cq.chapter_id
      WHERE cc.course_id = ?
      GROUP BY cc.id
      ORDER BY cc.chapter_number
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