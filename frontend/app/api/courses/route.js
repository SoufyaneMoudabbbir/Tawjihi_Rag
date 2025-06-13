import { NextResponse } from "next/server"
import { openDb } from "@/lib/db"
import { writeFile, mkdir } from "fs/promises"
import path from "path"

export async function POST(request) {
  try {
    const formData = await request.formData()
    const userId = formData.get('userId')
    const name = formData.get('name')
    const description = formData.get('description') || ''
    const professor = formData.get('professor') || ''
    const semester = formData.get('semester') || ''
    const files = formData.getAll('files')


    if (!userId || !name) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 })
    }
    const AnalysisProgressIndicator = ({ courseId, courseName }) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.9 }}
    animate={{ opacity: 1, scale: 1 }}
    className="absolute inset-0 bg-white/95 backdrop-blur-sm rounded-2xl flex flex-col items-center justify-center z-10"
  >
    <div className="relative">
      {/* Spinning Icon */}
      <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-200 border-t-blue-600 mb-4"></div>
      
      {/* Brain Icon in Center */}
      <div className="absolute inset-0 flex items-center justify-center">
        <motion.div
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <BookOpen className="h-6 w-6 text-blue-600" />
        </motion.div>
      </div>
    </div>
    
    <div className="text-center">
      <h4 className="font-semibold text-gray-900 mb-1">🧠 AI Analyzing Course</h4>
      <p className="text-sm text-gray-600 mb-2">Detecting chapters and structure...</p>
      <p className="text-xs text-gray-500">{courseName}</p>
    </div>

    {/* Progress Steps */}
    <div className="mt-4 flex space-x-2">
      {['📖 Reading', '🧠 Analyzing', '✨ Structuring'].map((step, index) => (
        <motion.div
          key={step}
          initial={{ opacity: 0.3 }}
          animate={{ 
            opacity: [0.3, 1, 0.3],
            scale: [1, 1.05, 1]
          }}
          transition={{ 
            duration: 1.5, 
            repeat: Infinity,
            delay: index * 0.5
          }}
          className="text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded-full"
        >
          {step}
        </motion.div>
      ))}
    </div>
  </motion.div>
)

    const db = await openDb()

    // Create courses table if it doesn't exist
    await db.exec(`
      CREATE TABLE IF NOT EXISTS courses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        professor TEXT,
        semester TEXT,
        status TEXT DEFAULT 'active',
        file_count INTEGER DEFAULT 0,
        chat_count INTEGER DEFAULT 0,
        progress INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_accessed DATETIME
      )
    `)

    // Create course_files table if it doesn't exist
    await db.exec(`
      CREATE TABLE IF NOT EXISTS course_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        original_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_size INTEGER,
        upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        processed BOOLEAN DEFAULT FALSE,
        FOREIGN KEY (course_id) REFERENCES courses (id) ON DELETE CASCADE
      )
    `)

    // Create indexes
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_courses_user_id ON courses(user_id)`)
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_course_files_course_id ON course_files(course_id)`)

    // Insert course
    const courseResult = await db.run(
      "INSERT INTO courses (user_id, name, description, professor, semester) VALUES (?, ?, ?, ?, ?)",
      [userId, name, description, professor, semester]
    )

    const courseId = courseResult.lastID

    // Handle file uploads
    let fileCount = 0
    if (files && files.length > 0) {
      // Create upload directory
      const uploadDir = path.join(process.cwd(), 'uploads', userId, courseId.toString())
      await mkdir(uploadDir, { recursive: true })

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
          await db.run(
            "INSERT INTO course_files (course_id, filename, original_name, file_path, file_size) VALUES (?, ?, ?, ?, ?)",
            [courseId, filename, file.name, filePath, buffer.length]
          )
          
          fileCount++
        }
      }

      // Update course file count
      await db.run("UPDATE courses SET file_count = ? WHERE id = ?", [fileCount, courseId])
      if (fileCount > 0) {
        try {
            const response = await fetch(`http://localhost:8000/courses/${courseId}/load`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
            })
            
            if (response.ok) {
            const result = await response.json()
            console.log(`Course ${courseId} materials processed:`, result)
            } else {
            console.error(`Failed to process course ${courseId} materials`)
            }
        } catch (error) {
            console.error('Error triggering PDF processing:', error)
            // Don't fail the course creation if backend is down
        }
        }
         try {
          console.log(`🧠 Starting auto-analysis for course ${courseId}`)
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
        // END OF NEW BLOCK ⬆️⬆️⬆️
        
    }

    return NextResponse.json({ 
      success: true, 
      courseId,
      message: `Course created successfully with ${fileCount} files uploaded` 
    })

  } catch (error) {
    console.error("Error creating course:", error)
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
      CREATE TABLE IF NOT EXISTS courses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        professor TEXT,
        semester TEXT,
        status TEXT DEFAULT 'active',
        file_count INTEGER DEFAULT 0,
        chat_count INTEGER DEFAULT 0,
        progress INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_accessed DATETIME
      )
    `)

    await db.exec(`
      CREATE TABLE IF NOT EXISTS course_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        original_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_size INTEGER,
        upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        processed BOOLEAN DEFAULT FALSE,
        FOREIGN KEY (course_id) REFERENCES courses (id) ON DELETE CASCADE
      )
    `)

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

    // Create indexes
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_courses_user_id ON courses(user_id)`)
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_course_files_course_id ON course_files(course_id)`)
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id)`)
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_chat_sessions_course_id ON chat_sessions(course_id)`)

    // Get courses with file and chat counts
    const courses = await db.all(`
      SELECT 
        c.*,
        COUNT(DISTINCT cf.id) as file_count,
        COUNT(DISTINCT cs.id) as chat_count
      FROM courses c
      LEFT JOIN course_files cf ON c.id = cf.course_id
      LEFT JOIN chat_sessions cs ON c.id = cs.course_id
      WHERE c.user_id = ?
      GROUP BY c.id
      ORDER BY c.updated_at DESC
    `, [userId])

    return NextResponse.json({ courses })

  } catch (error) {
    console.error("Error fetching courses:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}