import { NextResponse } from "next/server"
import { openDb } from "@/lib/db"
import { rm } from "fs/promises"
import path from "path"

export async function DELETE(request, { params }) {
  try {
    const courseId = params.courseId
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get("userId")

    if (!courseId || !userId) {
      return NextResponse.json({ error: "Missing required parameters" }, { status: 400 })
    }

    const db = await openDb()

    // Verify course belongs to user and get course info
    const course = await db.get("SELECT * FROM courses WHERE id = ? AND user_id = ?", [courseId, userId])
    if (!course) {
      return NextResponse.json({ error: "Course not found" }, { status: 404 })
    }

    // Get all files for this course
    const files = await db.all("SELECT file_path FROM course_files WHERE course_id = ?", [courseId])

    // Delete physical files
    try {
      const courseDir = path.join(process.cwd(), 'uploads', userId, courseId)
      await rm(courseDir, { recursive: true, force: true })
    } catch (fileError) {
      console.warn("Error deleting course files:", fileError)
      // Continue with database cleanup even if file deletion fails
    }

    // Delete from database (cascading will handle course_files)
    await db.run("DELETE FROM courses WHERE id = ?", [courseId])

    // Also delete related chat sessions
    await db.run("DELETE FROM chat_sessions WHERE course_id = ?", [courseId])

    return NextResponse.json({ 
      success: true, 
      message: "Course deleted successfully" 
    })

  } catch (error) {
    console.error("Error deleting course:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}