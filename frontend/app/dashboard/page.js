"use client"

import { useUser, UserButton } from "@clerk/nextjs"
import { motion } from "framer-motion"
import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { 
  Plus, 
  History, 
  MessageCircle, 
  BookOpen, 
  TrendingUp, 
  FileText,
  Target,
  Award,
  Calendar,
  Brain,
  Users,
  Clock
} from "lucide-react"
import ChatHistorySidebar from "@/components/ChatHistorySidebar"

export default function DashboardPage() {
  const { user, isLoaded } = useUser()
  const router = useRouter()
  const [hasCompletedForm, setHasCompletedForm] = useState(false)
  const [chatSessions, setChatSessions] = useState([])
  const [courses, setCourses] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [showHistory, setShowHistory] = useState(false)
  const [stats, setStats] = useState({
    totalCourses: 0,
    totalSessions: 0,
    totalFiles: 0,
    avgProgress: 0,
    studyStreak: 0,
    thisWeekSessions: 0
  })

  useEffect(() => {
    if (isLoaded && user) {
      loadData()
    }
  }, [isLoaded, user])

  const loadData = async () => {
    try {
      // Check form completion
      const formResponse = await fetch(`/api/responses?userId=${user.id}`)
      setHasCompletedForm(formResponse.ok)

      // Load courses
      const coursesResponse = await fetch(`/api/courses?userId=${user.id}`)
      if (coursesResponse.ok) {
        const coursesData = await coursesResponse.json()
        setCourses(coursesData.courses || [])
      }

      // Load chat sessions
      const sessionsResponse = await fetch(`/api/chats?userId=${user.id}`)
      if (sessionsResponse.ok) {
        const sessionsData = await sessionsResponse.json()
        setChatSessions(sessionsData.sessions || [])
      }

    } catch (error) {
      console.error("Error loading dashboard data:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const calculateStats = () => {
    const totalCourses = courses.length
    const totalSessions = chatSessions.length
    const totalFiles = courses.reduce((sum, course) => sum + (course.file_count || 0), 0)
    const avgProgress = courses.length > 0 
      ? Math.round(courses.reduce((sum, course) => sum + (course.progress || 0), 0) / courses.length)
      : 0

    // Calculate this week's sessions
    const oneWeekAgo = new Date()
    oneWeekAgo.setDate(oneWeekAgo.getDate() - 7)
    const thisWeekSessions = chatSessions.filter(session => 
      new Date(session.created_at) > oneWeekAgo
    ).length

    setStats({
      totalCourses,
      totalSessions,
      totalFiles,
      avgProgress,
      studyStreak: Math.min(thisWeekSessions, 7),
      thisWeekSessions
    })
  }

  useEffect(() => {
    if (courses.length > 0 || chatSessions.length > 0) {
      calculateStats()
    }
  }, [courses, chatSessions])

  const handleCreateNewPath = () => {
    router.push("/questionnaire")
  }

  const handleOpenChat = (sessionId) => {
    router.push(`/chat/${sessionId}`)
  }

  const handleDeleteSession = async (sessionId) => {
    try {
      const response = await fetch(`/api/chats/${sessionId}`, {
        method: "DELETE",
      })
      if (response.ok) {
        setChatSessions(chatSessions.filter((session) => session.id !== sessionId))
      }
    } catch (error) {
      console.error("Error deleting session:", error)
    }
  }

  const handleRenameSession = async (sessionId, newTitle) => {
    try {
      const response = await fetch(`/api/chats/${sessionId}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ title: newTitle }),
      })
      if (response.ok) {
        setChatSessions(
          chatSessions.map((session) => (session.id === sessionId ? { ...session, title: newTitle } : session)),
        )
      }
    } catch (error) {
      console.error("Error renaming session:", error)
    }
  }

  const startGeneralLearningSession = async () => {
    try {
      const response = await fetch('/api/chats', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          userId: user.id,
          title: `General Learning Session - ${new Date().toLocaleDateString()}`
        })
      })

      if (response.ok) {
        const data = await response.json()
        router.push(`/chat/${data.sessionId}`)
      }
    } catch (error) {
      console.error("Error starting learning session:", error)
    }
  }

  if (!isLoaded || isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your learning dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">EduBot Dashboard</h1>
                <p className="text-sm text-gray-500">Your personalized learning hub</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowHistory(!showHistory)}
                className="btn-secondary inline-flex items-center space-x-2"
              >
                <History className="h-4 w-4" />
                <span>History</span>
              </button>
              <UserButton afterSignOutUrl="/" />
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <ChatHistorySidebar
          isOpen={showHistory}
          sessions={chatSessions}
          onOpenChat={handleOpenChat}
          onDeleteSession={handleDeleteSession}
          onRenameSession={handleRenameSession}
          onClose={() => setShowHistory(false)}
        />

        {/* Main Content */}
        <div className={`flex-1 transition-all duration-300 ${showHistory ? "ml-80" : ""}`}>
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            
            {/* Welcome Section */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
              <h2 className="text-3xl font-bold text-gray-900 mb-2">
                Welcome back, {user?.firstName}! 🎓
              </h2>
              <p className="text-lg text-gray-600">
                {hasCompletedForm
                  ? "Ready to continue your learning journey?"
                  : "Let's set up your learning profile to get personalized educational support."}
              </p>
            </motion.div>

            {!hasCompletedForm ? (
              /* First-time user setup */
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-2xl mx-auto">
                <div className="bg-white rounded-2xl border border-gray-200 shadow-lg p-8 text-center">
                  <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl mb-6">
                    <BookOpen className="h-8 w-8 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-4">Set Up Your Learning Profile</h3>
                  <p className="text-gray-600 mb-6">
                    Tell us about your courses, learning style, and goals to get personalized AI tutoring.
                  </p>
                  <button 
                    onClick={handleCreateNewPath} 
                    className="btn-primary inline-flex items-center space-x-2"
                  >
                    <Plus className="h-5 w-5" />
                    <span>Complete Learning Profile</span>
                  </button>
                </div>
              </motion.div>
            ) : (
              <>
                {/* Learning Stats Grid */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
                  className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8"
                >
                  <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-xs text-gray-500 font-medium">COURSES</p>
                        <p className="text-2xl font-bold text-gray-900">{stats.totalCourses}</p>
                      </div>
                      <BookOpen className="h-8 w-8 text-blue-600" />
                    </div>
                  </div>
                  
                  <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-xs text-gray-500 font-medium">SESSIONS</p>
                        <p className="text-2xl font-bold text-gray-900">{stats.totalSessions}</p>
                      </div>
                      <MessageCircle className="h-8 w-8 text-green-600" />
                    </div>
                  </div>
                  
                  <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-xs text-gray-500 font-medium">MATERIALS</p>
                        <p className="text-2xl font-bold text-gray-900">{stats.totalFiles}</p>
                      </div>
                      <FileText className="h-8 w-8 text-purple-600" />
                    </div>
                  </div>
                  
                  <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-xs text-gray-500 font-medium">PROGRESS</p>
                        <p className="text-2xl font-bold text-gray-900">{stats.avgProgress}%</p>
                      </div>
                      <TrendingUp className="h-8 w-8 text-orange-600" />
                    </div>
                  </div>
                  
                  <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-xs text-gray-500 font-medium">STREAK</p>
                        <p className="text-2xl font-bold text-gray-900">{stats.studyStreak}</p>
                      </div>
                      <Award className="h-8 w-8 text-yellow-600" />
                    </div>
                  </div>
                  
                  <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-xs text-gray-500 font-medium">THIS WEEK</p>
                        <p className="text-2xl font-bold text-gray-900">{stats.thisWeekSessions}</p>
                      </div>
                      <Calendar className="h-8 w-8 text-indigo-600" />
                    </div>
                  </div>
                </motion.div>

                {/* Quick Actions */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="grid md:grid-cols-3 gap-6 mb-8"
                >
                  <button
                    onClick={() => router.push("/courses")}
                    className="bg-white rounded-2xl p-6 border border-gray-200 shadow-sm hover:shadow-lg transition-all duration-200 text-left group"
                  >
                    <div className="inline-flex items-center justify-center w-12 h-12 bg-blue-100 rounded-xl mb-4 group-hover:bg-blue-200 transition-colors">
                      <BookOpen className="h-6 w-6 text-blue-600" />
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">Manage Courses</h3>
                    <p className="text-gray-600 text-sm">Add courses, upload study materials, and organize your learning content</p>
                  </button>

                  <button
                    onClick={startGeneralLearningSession}
                    className="bg-white rounded-2xl p-6 border border-gray-200 shadow-sm hover:shadow-lg transition-all duration-200 text-left group"
                  >
                    <div className="inline-flex items-center justify-center w-12 h-12 bg-green-100 rounded-xl mb-4 group-hover:bg-green-200 transition-colors">
                      <MessageCircle className="h-6 w-6 text-green-600" />
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">Start Learning</h3>
                    <p className="text-gray-600 text-sm">Begin a learning session to get help with any subject or topic</p>
                  </button>

                  <button
                    onClick={handleCreateNewPath}
                    className="bg-white rounded-2xl p-6 border border-gray-200 shadow-sm hover:shadow-lg transition-all duration-200 text-left group"
                  >
                    <div className="inline-flex items-center justify-center w-12 h-12 bg-purple-100 rounded-xl mb-4 group-hover:bg-purple-200 transition-colors">
                      <Target className="h-6 w-6 text-purple-600" />
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">Update Profile</h3>
                    <p className="text-gray-600 text-sm">Refresh your learning preferences and academic goals</p>
                  </button>
                </motion.div>

                {/* Active Courses */}
                {courses.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="mb-8"
                  >
                    <div className="flex items-center justify-between mb-6">
                      <h3 className="text-xl font-semibold text-gray-900">Your Courses</h3>
                      <button
                        onClick={() => router.push("/courses")}
                        className="text-blue-600 hover:text-blue-700 text-sm font-medium"
                      >
                        View All →
                      </button>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {courses.slice(0, 6).map((course) => (
                        <motion.div
                          key={course.id}
                          whileHover={{ scale: 1.02 }}
                          className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm hover:shadow-md transition-all cursor-pointer"
                          onClick={() => router.push(`/courses`)}
                        >
                          <div className="flex items-start justify-between mb-3">
                            <h4 className="font-medium text-gray-900 line-clamp-2 flex-1">{course.name}</h4>
                            <div className="text-xs text-gray-500 ml-2 flex items-center">
                              <FileText className="h-3 w-3 mr-1" />
                              {course.file_count || 0}
                            </div>
                          </div>
                          
                          {course.professor && (
                            <p className="text-sm text-gray-600 mb-3">Prof. {course.professor}</p>
                          )}
                          
                          <div className="flex items-center justify-between text-xs text-gray-500 mb-3">
                            <span className="flex items-center">
                              <MessageCircle className="h-3 w-3 mr-1" />
                              {course.chat_count || 0} sessions
                            </span>
                            <span>{course.progress || 0}% complete</span>
                          </div>
                          
                          <div className="w-full bg-gray-200 rounded-full h-1.5">
                            <div 
                              className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                              style={{ width: `${course.progress || 0}%` }}
                            ></div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                )}

                {/* Recent Learning Sessions */}
                {chatSessions.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                  >
                    <div className="flex items-center justify-between mb-6">
                      <h3 className="text-xl font-semibold text-gray-900">Recent Learning Sessions</h3>
                      <button
                        onClick={() => setShowHistory(true)}
                        className="text-blue-600 hover:text-blue-700 text-sm font-medium"
                      >
                        View All →
                      </button>
                    </div>
                    <div className="grid gap-4 max-w-4xl">
                      {chatSessions.slice(0, 4).map((session) => (
                        <motion.div
                          key={session.id}
                          whileHover={{ scale: 1.01 }}
                          onClick={() => handleOpenChat(session.id)}
                          className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm hover:shadow-md transition-all cursor-pointer"
                        >
                          <div className="flex justify-between items-start">
                            <div className="flex-1">
                              <h4 className="font-medium text-gray-900 mb-1">{session.title}</h4>
                              <div className="flex items-center space-x-4 text-sm text-gray-500">
                                <span className="flex items-center">
                                  <BookOpen className="h-3 w-3 mr-1" />
                                  {session.courseName || 'General Learning'}
                                </span>
                                <span className="flex items-center">
                                  <MessageCircle className="h-3 w-3 mr-1" />
                                  {session.messageCount || 0} messages
                                </span>
                                <span className="flex items-center">
                                  <Clock className="h-3 w-3 mr-1" />
                                  {new Date(session.created_at).toLocaleDateString()}
                                </span>
                              </div>
                            </div>
                            <MessageCircle className="h-5 w-5 text-gray-400 ml-4" />
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                )}

                {/* Empty State */}
                {courses.length === 0 && chatSessions.length === 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="text-center py-12"
                  >
                    <Brain className="h-24 w-24 text-gray-300 mx-auto mb-6" />
                    <h3 className="text-xl font-semibold text-gray-900 mb-4">Ready to Start Learning?</h3>
                    <p className="text-gray-600 mb-8 max-w-md mx-auto">
                      Add your first course or start a learning session to begin your educational journey with AI assistance.
                    </p>
                    <div className="flex items-center justify-center space-x-4">
                      <button
                        onClick={() => router.push("/courses")}
                        className="btn-primary inline-flex items-center space-x-2"
                      >
                        <Plus className="h-5 w-5" />
                        <span>Add Your First Course</span>
                      </button>
                      <button
                        onClick={startGeneralLearningSession}
                        className="btn-secondary inline-flex items-center space-x-2"
                      >
                        <MessageCircle className="h-5 w-5" />
                        <span>Start Learning Session</span>
                      </button>
                    </div>
                  </motion.div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
