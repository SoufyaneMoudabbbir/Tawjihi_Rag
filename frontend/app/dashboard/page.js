"use client"

import { useUser, UserButton } from "@clerk/nextjs"
import { motion } from "framer-motion"
import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Plus, History, MessageCircle, BookOpen, TrendingUp } from "lucide-react"
import ChatHistorySidebar from "@/components/ChatHistorySidebar"

export default function DashboardPage() {
  const { user, isLoaded } = useUser()
  const router = useRouter()
  const [hasCompletedForm, setHasCompletedForm] = useState(false)
  const [chatSessions, setChatSessions] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [showHistory, setShowHistory] = useState(false)

  useEffect(() => {
    if (isLoaded && user) {
      checkFormCompletion()
      loadChatSessions()
    }
  }, [isLoaded, user])

  const checkFormCompletion = async () => {
    try {
      const response = await fetch(`/api/responses?userId=${user.id}`)
      if (response.ok) {
        setHasCompletedForm(true)
      } else {
        setHasCompletedForm(false)
      }
    } catch (error) {
      console.error("Error checking form completion:", error)
      setHasCompletedForm(false)
    } finally {
      setIsLoading(false)
    }
  }

  const loadChatSessions = async () => {
    try {
      const response = await fetch(`/api/chats?userId=${user.id}`)
      if (response.ok) {
        const data = await response.json()
        setChatSessions(data.sessions || [])
      }
    } catch (error) {
      console.error("Error loading chat sessions:", error)
    }
  }

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

  if (!isLoaded || isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <BookOpen className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-xl font-semibold text-gray-900">EduPath Dashboard</h1>
                <p className="text-sm text-gray-500">Your educational journey hub</p>
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
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-12">
              <h2 className="text-3xl font-bold text-gray-900 mb-4">Welcome back, {user?.firstName}!</h2>
              <p className="text-lg text-gray-600 mb-8">
                {hasCompletedForm
                  ? "Ready to explore new opportunities or continue your journey?"
                  : "Let's start by getting to know you better with our questionnaire."}
              </p>
            </motion.div>

            {!hasCompletedForm ? (
              /* First-time user flow */
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-2xl mx-auto">
                <div className="bg-blue-50 border border-blue-200 rounded-2xl p-8 text-center">
                  <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-6">
                    <BookOpen className="h-8 w-8 text-blue-600" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-4">Complete Your Profile</h3>
                  <p className="text-gray-600 mb-6">
                    Start by completing our questionnaire to get personalized educational guidance tailored to your
                    background and goals.
                  </p>
                  <button onClick={handleCreateNewPath} className="btn-primary inline-flex items-center space-x-2">
                    <Plus className="h-5 w-5" />
                    <span>Start Questionnaire</span>
                  </button>
                </div>
              </motion.div>
            ) : (
              /* Returning user flow */
              <>
                <div className="grid md:grid-cols-2 gap-6 max-w-2xl mx-auto mb-12">
                  <motion.button
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    onClick={handleCreateNewPath}
                    className="form-section text-center hover:shadow-xl transition-all duration-200 cursor-pointer group"
                  >
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-2xl mb-4 group-hover:bg-blue-200 transition-colors">
                      <Plus className="h-8 w-8 text-blue-600" />
                    </div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">Update Profile</h3>
                    <p className="text-gray-600">Refresh your preferences and explore new opportunities</p>
                  </motion.button>

                  <motion.button
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    onClick={() => setShowHistory(true)}
                    className="form-section text-center hover:shadow-xl transition-all duration-200 cursor-pointer group"
                  >
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-2xl mb-4 group-hover:bg-green-200 transition-colors">
                      <History className="h-8 w-8 text-green-600" />
                    </div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">View History</h3>
                    <p className="text-gray-600">Continue previous conversations and explorations</p>
                  </motion.button>
                </div>

                {/* Stats */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12"
                >
                  <div className="bg-white rounded-lg p-6 border border-gray-200">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-500">Total Sessions</p>
                        <p className="text-2xl font-bold text-gray-900">{chatSessions.length}</p>
                      </div>
                      <MessageCircle className="h-8 w-8 text-blue-600" />
                    </div>
                  </div>
                  <div className="bg-white rounded-lg p-6 border border-gray-200">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-500">Total Messages</p>
                        <p className="text-2xl font-bold text-gray-900">
                          {chatSessions.reduce((total, session) => total + (session.message_count || 0), 0)}
                        </p>
                      </div>
                      <TrendingUp className="h-8 w-8 text-green-600" />
                    </div>
                  </div>
                  <div className="bg-white rounded-lg p-6 border border-gray-200">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-gray-500">Profile Status</p>
                        <p className="text-sm font-semibold text-green-600">Complete</p>
                      </div>
                      <BookOpen className="h-8 w-8 text-purple-600" />
                    </div>
                  </div>
                </motion.div>

                {/* Recent Sessions */}
                {chatSessions.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                  >
                    <h3 className="text-xl font-semibold text-gray-900 mb-6">Recent Explorations</h3>
                    <div className="grid gap-4 max-w-2xl mx-auto">
                      {chatSessions.slice(0, 3).map((session) => (
                        <motion.div
                          key={session.id}
                          whileHover={{ scale: 1.02 }}
                          onClick={() => handleOpenChat(session.id)}
                          className="form-section cursor-pointer hover:shadow-lg transition-all duration-200 text-left"
                        >
                          <div className="flex justify-between items-start">
                            <div className="flex-1">
                              <h4 className="font-medium text-gray-900 mb-1">{session.title}</h4>
                              <p className="text-sm text-gray-500">
                                {new Date(session.created_at).toLocaleDateString("en-US", {
                                  year: "numeric",
                                  month: "short",
                                  day: "numeric",
                                  hour: "2-digit",
                                  minute: "2-digit",
                                })}
                              </p>
                              <p className="text-xs text-gray-400 mt-1">{session.message_count || 0} messages</p>
                            </div>
                            <MessageCircle className="h-5 w-5 text-gray-400" />
                          </div>
                        </motion.div>
                      ))}
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
