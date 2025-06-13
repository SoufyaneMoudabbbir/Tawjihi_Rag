import { NextResponse } from "next/server"
import { openDb } from "@/lib/db"
import { writeFile, mkdir } from "fs/promises"
import path from "path"

export async function POST(request) {
  try {
    const formData = await request.formData()
    const userId = formData.get('userId')
    const courseId = formData.get('courseId')
    const files = formData.getAll('files')

    if (!userId || !courseId) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 })
    }

    const db = await openDb()

    // Verify course belongs to user
    const course = await db.get("SELECT id FROM courses WHERE id = ? AND user_id = ?", [courseId, userId])
    if (!course) {
      return NextResponse.json({ error: "Course not found" }, { status: 404 })
    }

    // Create upload directory
    const uploadDir = path.join(process.cwd(), 'uploads', userId, courseId)
    await mkdir(uploadDir, { recursive: true })

    let uploadedCount = 0
    const uploadedFiles = []

    for (const file of files) {
      if (file.type === 'application/pdf') {
        const bytes = await file.arrayBuffer()
        const buffer = Buffer.from(bytes)
        
        // Generate unique filename
        const timestamp = Date.now()
        const filename = `${timestamp}_${file.name}`
        const filePath = path.join(uploadDir, filename)
        
        // Save file
        await writeFile(filePath, buffer)
        
        // Save file info to database
        const result = await db.run(
          "INSERT INTO course_files (course_id, filename, original_name, file_path, file_size) VALUES (?, ?, ?, ?, ?)",
          [courseId, filename, file.name, filePath, buffer.length]
        )
        
        uploadedFiles.push({
          id: result.lastID,
          filename: filename,
          originalName: file.name,
          size: buffer.length
        })
        
        uploadedCount++
      }
    }

    // Update course file count and last accessed
    await db.run(`
      UPDATE courses 
      SET file_count = (SELECT COUNT(*) FROM course_files WHERE course_id = ?),
          updated_at = CURRENT_TIMESTAMP,
          last_accessed = CURRENT_TIMESTAMP
      WHERE id = ?
    `, [courseId, courseId])

        if (uploadedCount > 0) {
        try {
            console.log(`🧠 Starting auto-analysis for course ${courseId} after file upload`)
            const analysisResponse = await fetch(`http://localhost:3000/api/courses/${courseId}/analyze`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ 
                userId: userId 
            })
            })
            
            if (analysisResponse.ok) {
            const analysisResult = await analysisResponse.json()
            console.log(`✅ Auto-analysis completed for course ${courseId}:`, analysisResult)
            } else {
            console.warn(`⚠️ Auto-analysis failed for course ${courseId}`)
            }
        } catch (analysisError) {
            console.error('Auto-analysis error:', analysisError)
        }
        }

    return NextResponse.json({ 
      success: true, 
      uploadedCount,
      files: uploadedFiles,
      message: `${uploadedCount} file(s) uploaded successfully` 
    })

  } catch (error) {
    console.error("Error uploading files:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}