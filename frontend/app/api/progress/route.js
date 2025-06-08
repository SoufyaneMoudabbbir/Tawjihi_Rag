import { NextResponse } from "next/server"
import { openDb } from "@/lib/db"

export async function POST(request) {
  try {
    const { userId, courseId, activityType, topicCovered, progressData } = await request.json()

    if (!userId || !activityType) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 })
    }

    const db = await openDb()

    // Create learning_progress table if it doesn't exist
    await db.exec(`
      CREATE TABLE IF NOT EXISTS learning_progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        course_id INTEGER,
        activity_type TEXT NOT NULL,
        topic_covered TEXT,
        progress_data TEXT,
        session_duration INTEGER,
        questions_asked INTEGER DEFAULT 0,
        concepts_learned TEXT,
        difficulty_level TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (course_id) REFERENCES courses (id) ON DELETE CASCADE
      )
    `)

    // Create learning_topics table for tracking individual topics
    await db.exec(`
      CREATE TABLE IF NOT EXISTS learning_topics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        course_id INTEGER,
        topic_name TEXT NOT NULL,
        understanding_level INTEGER DEFAULT 0,
        times_studied INTEGER DEFAULT 1,
        last_studied DATETIME DEFAULT CURRENT_TIMESTAMP,
        needs_review BOOLEAN DEFAULT FALSE,
        UNIQUE(user_id, course_id, topic_name),
        FOREIGN KEY (course_id) REFERENCES courses (id) ON DELETE CASCADE
      )
    `)

    // Create indexes
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_learning_progress_user_id ON learning_progress(user_id)`)
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_learning_progress_course_id ON learning_progress(course_id)`)
    await db.exec(`CREATE INDEX IF NOT EXISTS idx_learning_topics_user_course ON learning_topics(user_id, course_id)`)

    // Insert progress record
    const result = await db.run(`
      INSERT INTO learning_progress 
      (user_id, course_id, activity_type, topic_covered, progress_data, questions_asked, concepts_learned, difficulty_level) 
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `, [
      userId,
      courseId || null,
      activityType,
      topicCovered || null,
      JSON.stringify(progressData || {}),
      progressData?.questionsAsked || 0,
      JSON.stringify(progressData?.conceptsLearned || []),
      progressData?.difficultyLevel || 'medium'
    ])

    // Update or insert topic progress if topic is specified
    if (topicCovered && courseId) {
      await db.run(`
        INSERT INTO learning_topics (user_id, course_id, topic_name, understanding_level, times_studied, last_studied)
        VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id, course_id, topic_name) 
        DO UPDATE SET 
          times_studied = times_studied + 1,
          last_studied = CURRENT_TIMESTAMP,
          understanding_level = CASE 
            WHEN ? > understanding_level THEN ?
            ELSE understanding_level 
          END
      `, [
        userId,
        courseId,
        topicCovered,
        progressData?.understandingLevel || 3,
        progressData?.understandingLevel || 3
      ])

      // Update course progress
      await updateCourseProgress(db, courseId, userId)
    }

    return NextResponse.json({ 
      success: true, 
      progressId: result.lastID,
      message: "Progress tracked successfully" 
    })

  } catch (error) {
    console.error("Error tracking progress:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get("userId")
    const courseId = searchParams.get("courseId")
    const timeframe = searchParams.get("timeframe") || "week" // week, month, all

    if (!userId) {
      return NextResponse.json({ error: "User ID required" }, { status: 400 })
    }

    const db = await openDb()

    // Calculate date filter
    let dateFilter = ""
    const params = [userId]
    
    if (timeframe === "week") {
      dateFilter = "AND created_at >= date('now', '-7 days')"
    } else if (timeframe === "month") {
      dateFilter = "AND created_at >= date('now', '-30 days')"
    }

    // Add course filter if specified
    let courseFilter = ""
    if (courseId) {
      courseFilter = "AND course_id = ?"
      params.push(courseId)
    }

    // Get overall progress stats
    const progressStats = await db.get(`
      SELECT 
        COUNT(*) as total_sessions,
        SUM(questions_asked) as total_questions,
        COUNT(DISTINCT topic_covered) as topics_covered,
        AVG(session_duration) as avg_session_duration
      FROM learning_progress 
      WHERE user_id = ? ${courseFilter} ${dateFilter}
    `, params)

    // Get learning topics with progress
    const topicsQuery = courseId 
      ? "SELECT * FROM learning_topics WHERE user_id = ? AND course_id = ? ORDER BY last_studied DESC"
      : "SELECT * FROM learning_topics WHERE user_id = ? ORDER BY last_studied DESC"
    
    const topicsParams = courseId ? [userId, courseId] : [userId]
    const topics = await db.all(topicsQuery, topicsParams)

    // Get recent learning activities
    const activities = await db.all(`
      SELECT 
        lp.*,
        c.name as course_name
      FROM learning_progress lp
      LEFT JOIN courses c ON lp.course_id = c.id
      WHERE lp.user_id = ? ${courseFilter} ${dateFilter}
      ORDER BY lp.created_at DESC
      LIMIT 20
    `, params)

    // Get daily progress for charts
    const dailyProgress = await db.all(`
      SELECT 
        DATE(created_at) as date,
        COUNT(*) as session_count,
        SUM(questions_asked) as questions_count,
        COUNT(DISTINCT topic_covered) as topics_count
      FROM learning_progress 
      WHERE user_id = ? ${courseFilter} ${dateFilter}
      GROUP BY DATE(created_at)
      ORDER BY date DESC
    `, params)

    // Calculate learning streaks
    const streak = await calculateLearningStreak(db, userId, courseId)

    return NextResponse.json({
      stats: {
        ...progressStats,
        learning_streak: streak.current,
        longest_streak: streak.longest,
        topics_mastered: topics.filter(t => t.understanding_level >= 4).length,
        topics_need_review: topics.filter(t => t.needs_review).length
      },
      topics,
      activities: activities.map(activity => ({
        ...activity,
        progress_data: activity.progress_data ? JSON.parse(activity.progress_data) : {},
        concepts_learned: activity.concepts_learned ? JSON.parse(activity.concepts_learned) : []
      })),
      dailyProgress,
      timeframe
    })

  } catch (error) {
    console.error("Error fetching progress:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

async function updateCourseProgress(db, courseId, userId) {
  try {
    // Calculate course progress based on topics understanding
    const topicsStats = await db.get(`
      SELECT 
        COUNT(*) as total_topics,
        AVG(understanding_level) as avg_understanding,
        COUNT(CASE WHEN understanding_level >= 4 THEN 1 END) as mastered_topics
      FROM learning_topics 
      WHERE course_id = ? AND user_id = ?
    `, [courseId, userId])

    if (topicsStats.total_topics > 0) {
      const progress = Math.round((topicsStats.avg_understanding / 5) * 100)
      await db.run(`
        UPDATE courses 
        SET progress = ?, last_accessed = CURRENT_TIMESTAMP 
        WHERE id = ? AND user_id = ?
      `, [progress, courseId, userId])
    }
  } catch (error) {
    console.error("Error updating course progress:", error)
  }
}

async function calculateLearningStreak(db, userId, courseId = null) {
  try {
    const courseFilter = courseId ? "AND course_id = ?" : ""
    const params = courseId ? [userId, courseId] : [userId]

    // Get distinct learning days
    const learningDays = await db.all(`
      SELECT DISTINCT DATE(created_at) as learning_date
      FROM learning_progress 
      WHERE user_id = ? ${courseFilter}
      ORDER BY learning_date DESC
    `, params)

    if (learningDays.length === 0) {
      return { current: 0, longest: 0 }
    }

    let currentStreak = 0
    let longestStreak = 0
    let tempStreak = 0
    let previousDate = null

    for (const day of learningDays) {
      const currentDate = new Date(day.learning_date)
      
      if (previousDate === null) {
        // First day
        tempStreak = 1
        currentStreak = 1
      } else {
        const dayDiff = (previousDate - currentDate) / (1000 * 60 * 60 * 24)
        
        if (dayDiff === 1) {
          // Consecutive day
          tempStreak++
          if (previousDate.toDateString() === new Date().toDateString() || 
              currentDate.toDateString() === new Date().toDateString()) {
            currentStreak = tempStreak
          }
        } else {
          // Streak broken
          longestStreak = Math.max(longestStreak, tempStreak)
          tempStreak = 1
          if (currentDate.toDateString() === new Date().toDateString()) {
            currentStreak = 1
          } else {
            currentStreak = 0
          }
        }
      }
      
      previousDate = currentDate
    }

    longestStreak = Math.max(longestStreak, tempStreak)

    return { current: currentStreak, longest: longestStreak }
  } catch (error) {
    console.error("Error calculating streak:", error)
    return { current: 0, longest: 0 }
  }
}