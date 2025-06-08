"use client"

import { useUser } from "@clerk/nextjs"
import { motion } from "framer-motion"
import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { 
  ArrowLeft, 
  Upload, 
  BookOpen, 
  FileText, 
  Trash2, 
  Plus,
  MessageCircle,
  Clock,
  CheckCircle,
  AlertCircle,
  Edit3,
  Calendar
} from "lucide-react"

export default function CoursesPage() {
  const { user, isLoaded } = useUser()
  const router = useRouter()
  const [courses, setCourses] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [uploadingCourse, setUploadingCourse] = useState(null)
  const [showAddCourse, setShowAddCourse] = useState(false)
  const [newCourse, setNewCourse] = useState({
    name: "",
    description: "",
    professor: "",
    semester: "",
    files: []
  })

  useEffect(() => {
    if (isLoaded && user) {
      loadCourses()
    }
  }, [isLoaded, user])

  const loadCourses = async () => {
    try {
      const response = await fetch(`/api/courses?userId=${user.id}`)
      if (response.ok) {
        const data = await response.json()
        setCourses(data.courses || [])
      }
    } catch (error) {
      console.error("Error loading courses:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileUpload = async (event, courseId = null) => {
    const files = Array.from(event.target.files)
    const pdfFiles = files.filter(file => file.type === 'application/pdf')
    
    if (pdfFiles.length === 0) {
      alert("Please select PDF files only.")
      return
    }

    if (!courseId) {
      setNewCourse(prev => ({
        ...prev,
        files: [...prev.files, ...pdfFiles]
      }))
      return
    }

    setUploadingCourse(courseId)
    
    const formData = new FormData()
    formData.append('userId', user.id)
    formData.append('courseId', courseId)
    pdfFiles.forEach(file => {
      formData.append('files', file)
    })

    try {
      const response = await fetch('/api/courses/upload', {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        await loadCourses()
        alert(`Successfully uploaded ${pdfFiles.length} file(s)`)
      } else {
        alert("Failed to upload files")
      }
    } catch (error) {
      console.error("Error uploading files:", error)
      alert("Error uploading files")
    } finally {
      setUploadingCourse(null)
    }
  }

  const handleCreateCourse = async () => {
    if (!newCourse.name.trim()) {
      alert("Please enter a course name")
      return
    }

    const formData = new FormData()
    formData.append('userId', user.id)
    formData.append('name', newCourse.name)
    formData.append('description', newCourse.description)
    formData.append('professor', newCourse.professor)
    formData.append('semester', newCourse.semester)
    
    newCourse.files.forEach(file => {
      formData.append('files', file)
    })

    try {
      const response = await fetch('/api/courses', {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        await loadCourses()
        setNewCourse({ name: "", description: "", professor: "", semester: "", files: [] })
        setShowAddCourse(false)
        alert("Course created successfully!")
      } else {
        alert("Failed to create course")
      }
    } catch (error) {
      console.error("Error creating course:", error)
      alert("Error creating course")
    }
  }

  const handleDeleteCourse = async (courseId) => {
    if (!confirm("Are you sure you want to delete this course and all its materials?")) {
      return
    }

    try {
      const response = await fetch(`/api/courses/${courseId}?userId=${user.id}`, {
        method: 'DELETE'
      })

      if (response.ok) {
        await loadCourses()
        alert("Course deleted successfully")
      } else {
        alert("Failed to delete course")
      }
    } catch (error) {
      console.error("Error deleting course:", error)
      alert("Error deleting course")
    }
  }

  const startLearningSession = async (courseId, courseName) => {
    try {
      const response = await fetch('/api/chats', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          userId: user.id,
          title: `${courseName} Study Session - ${new Date().toLocaleDateString()}`,
          courseId: courseId
        })
      })

      if (response.ok) {
        const data = await response.json()
        router.push(`/chat/${data.sessionId}`)
      } else {
        alert("Failed to start learning session")
      }
    } catch (error) {
      console.error("Error starting session:", error)
      alert("Error starting learning session")
    }
  }

  if (!isLoaded || isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your courses...</p>
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
              <button
                onClick={() => router.push("/dashboard")}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <ArrowLeft className="h-5 w-5 text-gray-600" />
              </button>
              <BookOpen className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-xl font-semibold text-gray-900">My Courses</h1>
                <p className="text-sm text-gray-500">Manage your course materials and learning content</p>
              </div>
            </div>
            <button
              onClick={() => setShowAddCourse(true)}
              className="btn-primary inline-flex items-center space-x-2"
            >
              <Plus className="h-4 w-4" />
              <span>Add Course</span>
            </button>
          </div>
          </div>
        
      </header>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Add Course Modal */}
        {showAddCourse && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="bg-white rounded-2xl p-6 w-full max-w-md max-h-[90vh] overflow-y-auto"
            >
              <h2 className="text-xl font-bold text-gray-900 mb-4">Add New Course</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Course Name *</label>
                  <input
                    type="text"
                    placeholder="e.g., Introduction to Computer Science"
                    value={newCourse.name}
                    onChange={(e) => setNewCourse(prev => ({ ...prev, name: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Professor/Instructor</label>
                  <input
                    type="text"
                    placeholder="e.g., Dr. Jane Smith"
                    value={newCourse.professor}
                    onChange={(e) => setNewCourse(prev => ({ ...prev, professor: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Semester/Term</label>
                  <input
                    type="text"
                    placeholder="e.g., Fall 2024"
                    value={newCourse.semester}
                    onChange={(e) => setNewCourse(prev => ({ ...prev, semester: e.target.value }))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Course Description</label>
                  <textarea
                    placeholder="Brief description of the course content and objectives..."
                    value={newCourse.description}
                    onChange={(e) => setNewCourse(prev => ({ ...prev, description: e.target.value }))}
                    rows="3"
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Upload Course Materials (PDF files)
                  </label>
                  <label className="w-full border-2 border-dashed border-gray-300 rounded-xl p-6 text-center cursor-pointer hover:border-blue-400 transition-colors">
                    <Upload className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                    <span className="text-sm text-gray-600">
                      Click to upload PDFs or drag and drop
                    </span>
                    <input
                      type="file"
                      multiple
                      accept=".pdf"
                      onChange={(e) => handleFileUpload(e)}
                      className="hidden"
                    />
                  </label>
                  {newCourse.files.length > 0 && (
                    <div className="mt-2">
                      <p className="text-sm text-gray-600 mb-2">
                        {newCourse.files.length} file(s) selected:
                      </p>
                      <div className="space-y-1">
                        {newCourse.files.map((file, index) => (
                          <div key={index} className="flex items-center justify-between text-xs bg-gray-50 p-2 rounded">
                            <span className="truncate">{file.name}</span>
                            <button
                              onClick={() => setNewCourse(prev => ({
                                ...prev,
                                files: prev.files.filter((_, i) => i !== index)
                              }))}
                              className="text-red-500 hover:text-red-700 ml-2"
                            >
                              ×
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
              
              <div className="flex space-x-4 mt-6">
                <button
                  onClick={() => {
                    setShowAddCourse(false)
                    setNewCourse({ name: "", description: "", professor: "", semester: "", files: [] })
                  }}
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-xl text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreateCourse}
                  disabled={!newCourse.name.trim()}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                >
                  Create Course
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}

        {/* Courses Grid */}
        {courses.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center py-16"
          >
            <div className="w-24 h-24 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <BookOpen className="h-12 w-12 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">No Courses Yet</h2>
            <p className="text-gray-600 mb-8 max-w-md mx-auto">
              Get started by adding your first course. Upload your syllabus, lecture notes, and textbooks to begin learning with AI assistance.
            </p>
            <button
              onClick={() => setShowAddCourse(true)}
              className="btn-primary inline-flex items-center space-x-2"
            >
              <Plus className="h-5 w-5" />
              <span>Add Your First Course</span>
            </button>
          </motion.div>
        ) : (
          <>
            {/* Course Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8"
            >
              <div className="bg-white rounded-xl p-4 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-500">Total Courses</p>
                    <p className="text-2xl font-bold text-gray-900">{courses.length}</p>
                  </div>
                  <BookOpen className="h-8 w-8 text-blue-600" />
                </div>
              </div>
              
              <div className="bg-white rounded-xl p-4 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-500">Total Materials</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {courses.reduce((sum, course) => sum + (course.file_count || 0), 0)}
                    </p>
                  </div>
                  <FileText className="h-8 w-8 text-green-600" />
                </div>
              </div>
              
              <div className="bg-white rounded-xl p-4 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-500">Learning Sessions</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {courses.reduce((sum, course) => sum + (course.chat_count || 0), 0)}
                    </p>
                  </div>
                  <MessageCircle className="h-8 w-8 text-purple-600" />
                </div>
              </div>
              
              <div className="bg-white rounded-xl p-4 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-500">Avg Progress</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {Math.round(courses.reduce((sum, course) => sum + (course.progress || 0), 0) / courses.length)}%
                    </p>
                  </div>
                  <CheckCircle className="h-8 w-8 text-orange-600" />
                </div>
              </div>
            </motion.div>

            {/* Courses Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {courses.map((course, index) => (
                <motion.div
                  key={course.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6 hover:shadow-xl transition-all duration-300"
                >
                  {/* Course Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-gray-900 mb-2 line-clamp-2">
                        {course.name}
                      </h3>
                      {course.professor && (
                        <p className="text-sm text-gray-600 mb-1 flex items-center">
                          <Edit3 className="h-3 w-3 mr-1" />
                          Prof. {course.professor}
                        </p>
                      )}
                      {course.semester && (
                        <p className="text-sm text-gray-500 flex items-center">
                          <Calendar className="h-3 w-3 mr-1" />
                          {course.semester}
                        </p>
                      )}
                    </div>
                    <button
                      onClick={() => handleDeleteCourse(course.id)}
                      className="p-2 text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>

                  {/* Course Description */}
                  {course.description && (
                    <p className="text-gray-600 text-sm mb-4 line-clamp-3">
                      {course.description}
                    </p>
                  )}

                  {/* Course Stats */}
                  <div className="flex items-center justify-between mb-4 text-sm text-gray-500">
                    <div className="flex items-center space-x-4">
                      <div className="flex items-center space-x-1">
                        <FileText className="h-4 w-4" />
                        <span>{course.file_count || 0} files</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <MessageCircle className="h-4 w-4" />
                        <span>{course.chat_count || 0} sessions</span>
                      </div>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Clock className="h-4 w-4" />
                      <span>{course.last_accessed ? 'Recent' : 'New'}</span>
                    </div>
                  </div>

                  {/* Progress Bar */}
                  <div className="mb-4">
                    <div className="flex justify-between text-sm text-gray-600 mb-2">
                      <span>Learning Progress</span>
                      <span>{course.progress || 0}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-blue-500 to-indigo-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${course.progress || 0}%` }}
                      ></div>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="space-y-3">
                    <button
                      onClick={() => startLearningSession(course.id, course.name)}
                      className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 px-4 rounded-xl hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 inline-flex items-center justify-center space-x-2 shadow-md hover:shadow-lg"
                    >
                      <MessageCircle className="h-4 w-4" />
                      <span>Start Learning Session</span>
                    </button>
                    
                    <div className="flex space-x-2">
                      <label className="flex-1">
                        <input
                          type="file"
                          multiple
                          accept=".pdf"
                          onChange={(e) => handleFileUpload(e, course.id)}
                          className="hidden"
                          disabled={uploadingCourse === course.id}
                        />
                        <div className="w-full bg-gray-100 text-gray-700 py-2 px-4 rounded-xl hover:bg-gray-200 transition-colors cursor-pointer inline-flex items-center justify-center space-x-2">
                          {uploadingCourse === course.id ? (
                            <>
                              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                              <span className="text-sm">Uploading...</span>
                            </>
                          ) : (
                            <>
                              <Upload className="h-4 w-4" />
                              <span className="text-sm">Add Materials</span>
                            </>
                          )}
                        </div>
                      </label>
                      
                      <button
                        onClick={() => router.push(`/courses/${course.id}/files`)}
                        className="bg-gray-100 text-gray-700 py-2 px-4 rounded-xl hover:bg-gray-200 transition-colors inline-flex items-center justify-center"
                      >
                        <FileText className="h-4 w-4" />
                      </button>
                    </div>
                  </div>

                  {/* Course Status */}
                  <div className="mt-4 pt-4 border-t border-gray-100">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center space-x-1">
                        {course.status === 'active' ? (
                          <>
                            <CheckCircle className="h-3 w-3 text-green-500" />
                            <span className="text-green-600 font-medium">Active</span>
                          </>
                        ) : (
                          <>
                            <AlertCircle className="h-3 w-3 text-amber-500" />
                            <span className="text-amber-600 font-medium">Inactive</span>
                          </>
                        )}
                      </div>
                      <span className="text-gray-500">
                        Added {course.created_at ? new Date(course.created_at).toLocaleDateString() : 'recently'}
                      </span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </>
        )}

        {/* Learning Tips */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-12 bg-white/60 backdrop-blur-sm rounded-2xl p-6 border border-gray-200"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <BookOpen className="h-5 w-5 mr-2 text-blue-600" />
            How to Get the Most from Your Courses
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-semibold text-xs">
                1
              </div>
              <div>
                <p className="font-medium text-gray-900 mb-1">Upload Quality Materials</p>
                <p className="text-gray-600">Add lecture notes, textbooks, slides, and assignments. The more context you provide, the better your AI tutor can help you.</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 bg-green-100 text-green-600 rounded-full flex items-center justify-center font-semibold text-xs">
                2
              </div>
              <div>
                <p className="font-medium text-gray-900 mb-1">Ask Specific Questions</p>
                <p className="text-gray-600">Instead of "help me study," try "explain photosynthesis" or "create practice problems for calculus derivatives."</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center font-semibold text-xs">
                3
              </div>
              <div>
                <p className="font-medium text-gray-900 mb-1">Regular Learning Sessions</p>
                <p className="text-gray-600">Consistent, shorter study sessions are more effective than cramming. Use the AI tutor regularly to reinforce learning.</p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}