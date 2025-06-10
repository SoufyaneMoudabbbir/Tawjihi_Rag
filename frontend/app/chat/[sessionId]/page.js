"use client"

import { useUser, UserButton } from "@clerk/nextjs"
import { motion, AnimatePresence } from "framer-motion"
import { useState, useEffect, useRef } from "react"
import { useRouter, useParams } from "next/navigation"
import {
  Send,
  Bot,
  User,
  Loader,
  BookOpen,
  FileText,
  Menu,
  X,
  Home,
  ArrowLeft,
  Plus,
  ChevronRight,
  Target,
  Brain,
  MessageCircle,
  Clock,
  Edit2,      // ADD THIS
  Trash2,     // ADD THIS
  Check,      // ADD THIS
  X as XIcon  // ADD THIS
} from "lucide-react"
import Link from "next/link"

export default function ChatSessionPage() {
  const { user, isLoaded } = useUser()
  const router = useRouter()
  const params = useParams()
  const sessionId = params.sessionId
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)

  const [messages, setMessages] = useState([])
  const [inputMessage, setInputMessage] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [sessionTitle, setSessionTitle] = useState("")
  const [courseInfo, setCourseInfo] = useState(null)
  const [suggestedQuestions, setSuggestedQuestions] = useState([])
  const [userProfile, setUserProfile] = useState(null)
  const [showHistory, setShowHistory] = useState(false)
  const [chatSessions, setChatSessions] = useState([])
  const [editingSessionId, setEditingSessionId] = useState(null)
  const [editingTitle, setEditingTitle] = useState("")
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)


  useEffect(() => {
    if (isLoaded && user && sessionId) {
      loadChatSession()
      loadChatSessions()
    }
  }, [isLoaded, user, sessionId])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    adjustTextareaHeight()
  }, [inputMessage])

  const adjustTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
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

  const loadChatSession = async () => {
    try {
      const response = await fetch(`/api/chats/${sessionId}`)
      if (response.ok) {
        const data = await response.json()
        setMessages(data.messages || [])
        setSessionTitle(data.title || "Learning Session")
        setCourseInfo(data.course || null)
        
        // Generate suggested questions based on course context
        if (data.course) {
          generateSuggestedQuestions(data.course)
        } else {
          generateGeneralSuggestedQuestions()
        }

        // Load user profile
        const profileResponse = await fetch(`/api/responses?userId=${user.id}`)
        if (profileResponse.ok) {
          const profileData = await profileResponse.json()
          setUserProfile(profileData.responses || {})
          console.log('User profile loaded:', profileData.responses)
        }
      } else {
        router.push("/dashboard")
      }
    } catch (error) {
      console.error("Error loading chat session:", error)
      router.push("/dashboard")
    } finally {
      setIsLoading(false)
    }
  }

  const generateSuggestedQuestions = (course) => {
    const questions = [
      `Explain the key concepts in ${course.name}`,
      `Create a study plan for ${course.name}`,
      `What are common exam questions for this course?`,
      `Generate practice problems for ${course.name}`
    ]
    setSuggestedQuestions(questions)
  }

  const generateGeneralSuggestedQuestions = () => {
    const questions = [
      "Help me create a study schedule",
      "Explain a concept I'm struggling with",
      "What are effective study techniques?",
      "How can I improve my note-taking?"
    ]
    setSuggestedQuestions(questions)
  }

  const handleSendMessage = async (messageText = null) => {
    const message = messageText || inputMessage.trim()
    if (!message || isTyping || isStreaming) return

    const userMessage = {
      id: Date.now(),
      type: "user",
      content: message,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInputMessage("")
    setIsTyping(true)
    setIsStreaming(true)

    // Save user message to database
    try {
      await fetch(`/api/chats/${sessionId}/messages`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage,
        }),
      })
    } catch (error) {
      console.error("Error saving message:", error)
    }

    // Call the educational RAG API
    try {
      const response = await fetch('http://localhost:8000/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: message,
          course_id: courseInfo?.id || null,
          user_id: user.id,
          user_profile: userProfile,
          stream: true
        }),
      })

      if (response.ok) {
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let botResponse = ""
        let responseId = Date.now() + 1
        
        // Add initial bot message
        setMessages((prev) => [...prev, {
          id: responseId,
          type: "bot",
          content: "",
          timestamp: new Date(),
        }])

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value)
          const lines = chunk.split('\n')
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6))
                
                if (data.type === 'content') {
                  botResponse += data.data
                  setMessages((prev) => 
                    prev.map(msg => 
                      msg.id === responseId 
                        ? { ...msg, content: botResponse }
                        : msg
                    )
                  )
                } else if (data.type === 'done') {
                  // Save bot response to database
                  try {
                    await fetch(`/api/chats/${sessionId}/messages`, {
                      method: "POST",
                      headers: {
                        "Content-Type": "application/json",
                      },
                      body: JSON.stringify({
                        message: {
                          id: responseId,
                          type: "bot",
                          content: botResponse,
                          timestamp: new Date(),
                        },
                      }),
                    })
                  } catch (error) {
                    console.error("Error saving bot response:", error)
                  }
                }
              } catch (parseError) {
                console.error("Error parsing SSE data:", parseError)
              }
            }
          }
        }
      } else {
        // Fallback response
        const fallbackResponse = {
          id: Date.now() + 1,
          type: "bot",
          content: courseInfo 
            ? `I'm here to help you learn ${courseInfo.name}. Could you please rephrase your question or ask about a specific topic from your course materials?`
            : "I'm here to help with your studies. Could you please provide more details about what you'd like to learn or ask about?",
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, fallbackResponse])
      }
    } catch (error) {
      console.error("Error calling RAG API:", error)
      
      // Fallback response for network errors
      const errorResponse = {
        id: Date.now() + 1,
        type: "bot",
        content: "I'm having trouble connecting right now. Please try again in a moment.",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorResponse])
    } finally {
      setIsTyping(false)
      setIsStreaming(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleOpenChat = (session) => {
    router.push(`/chat/${session.id}`)
  }

  const createNewChat = async () => {
    try {
      const response = await fetch('/api/chats', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          userId: user.id,
          title: `New Learning Session - ${new Date().toLocaleDateString()}`
        })
      })

      if (response.ok) {
        const data = await response.json()
        router.push(`/chat/${data.sessionId}`)
      }
    } catch (error) {
      console.error("Error creating chat:", error)
    }
  }

  if (!isLoaded || isLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <Loader className="h-8 w-8 animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading your learning session...</p>
        </div>
      </div>
    )
  }
  const handleRenameSession = (sessionId, currentTitle) => {
      setEditingSessionId(sessionId)
      setEditingTitle(currentTitle)
    }

    const handleSaveRename = async (sessionId) => {
      if (!editingTitle.trim()) return
      
      try {
        const response = await fetch(`/api/chats/${sessionId}`, {
          method: 'PATCH',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            title: editingTitle.trim()
          })
        })
        
        if (response.ok) {
          setChatSessions(prev => 
            prev.map(session => 
              session.id === sessionId 
                ? { ...session, title: editingTitle.trim() }
                : session
            )
          )
          setEditingSessionId(null)
          setEditingTitle("")
        }
      } catch (error) {
        console.error("Error renaming session:", error)
      }
    }

    const handleCancelRename = () => {
      setEditingSessionId(null)
      setEditingTitle("")
    }

    const handleDeleteSession = async (sessionId) => {
      if (!confirm("Are you sure you want to delete this chat session?")) return
      
      try {
        const response = await fetch(`/api/chats/${sessionId}`, {
          method: 'DELETE'
        })
        
        if (response.ok) {
          setChatSessions(prev => prev.filter(session => session.id !== sessionId))
          
          // If we're deleting the current session, redirect to dashboard
          if (sessionId === parseInt(sessionId)) {
            router.push('/dashboard')
          }
        }
      } catch (error) {
        console.error("Error deleting session:", error)
      }
    }

  return (
    <div className="h-screen flex bg-gray-50">
      {/* Sidebar - Always visible on desktop, toggleable on mobile */}
      <aside className={`
  ${showHistory ? 'translate-x-0' : '-translate-x-full'} 
  lg:translate-x-0 
  transition-all duration-300 ease-in-out
  fixed lg:relative h-full bg-white shadow-lg z-50 flex flex-col border-r border-gray-200
  ${sidebarCollapsed ? 'w-16' : 'w-80'}
`}>
  {/* Sidebar Header */}
  <div className={`p-4 border-b border-gray-200 ${sidebarCollapsed ? 'px-2' : ''}`}>
    <div className="flex items-center justify-between mb-4">
      {!sidebarCollapsed && (
        <h2 className="text-lg font-semibold text-gray-900">Chats</h2>
      )}
      <div className="flex items-center space-x-2">
        <button
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors hidden lg:block"
          title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <Menu className="h-4 w-4 text-gray-600" />
        </button>
        <button
          onClick={() => setShowHistory(false)}
          className="lg:hidden p-2 hover:bg-gray-100 rounded-lg"
        >
          <X className="h-4 w-4 text-gray-500" />
        </button>
      </div>
    </div>
    
    <button
      onClick={createNewChat}
      className={`w-full flex items-center justify-center space-x-2 py-3 px-4 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-xl hover:from-blue-700 hover:to-blue-800 transition-all duration-200 shadow-lg hover:shadow-xl ${
        sidebarCollapsed ? 'px-2' : ''
      }`}
      title="New chat"
    >
      <Plus className="h-4 w-4" />
      {!sidebarCollapsed && <span className="font-medium">New Chat</span>}
    </button>
  </div>

  {/* Chat History */}
  <div className="flex-1 overflow-y-auto p-2 space-y-1">
    {chatSessions.map((session) => (
      <div key={session.id} className="relative group">
        <button
          onClick={() => handleOpenChat(session)}
          className={`
            w-full text-left p-3 rounded-xl transition-all duration-200 relative
            ${session.id === parseInt(sessionId) 
              ? 'bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 shadow-sm' 
              : 'hover:bg-gray-50 border border-transparent hover:border-gray-200'
            }
          `}
          title={sidebarCollapsed ? session.title : ''}
        >
          <div className="flex items-start space-x-3">
            <div className={`p-2 rounded-lg flex-shrink-0 ${
              session.id === parseInt(sessionId) 
                ? 'bg-blue-100 text-blue-600' 
                : 'bg-gray-100 text-gray-500 group-hover:bg-gray-200'
            }`}>
              <MessageCircle className="h-4 w-4" />
            </div>
            
            {!sidebarCollapsed && (
              <div className="flex-1 min-w-0">
                {editingSessionId === session.id ? (
                  <div className="flex items-center space-x-1" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="text"
                      value={editingTitle}
                      onChange={(e) => setEditingTitle(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          handleSaveRename(session.id);
                        } else if (e.key === 'Escape') {
                          handleCancelRename();
                        }
                      }}
                      className="flex-1 text-sm px-2 py-1 border border-blue-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
                      autoFocus
                    />
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleSaveRename(session.id);
                      }}
                      className="p-1 text-green-600 hover:bg-green-100 rounded-md transition-colors"
                    >
                      <Check className="h-3 w-3" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleCancelRename();
                      }}
                      className="p-1 text-red-600 hover:bg-red-100 rounded-md transition-colors"
                    >
                      <XIcon className="h-3 w-3" />
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="flex items-center justify-between mb-1">
                      <p className="text-sm font-medium text-gray-900 truncate pr-2">
                        {session.title}
                      </p>
                      {session.messageCount > 0 && (
                        <span className={`text-xs px-2 py-0.5 rounded-full flex-shrink-0 ${
                          session.id === parseInt(sessionId)
                            ? 'bg-blue-200 text-blue-800'
                            : 'bg-gray-200 text-gray-600'
                        }`}>
                          {session.messageCount}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <span className="truncate pr-2">
                        {session.courseName || 'General Learning'}
                      </span>
                      <span className="flex-shrink-0">
                        {new Date(session.updated_at || session.created_at).toLocaleDateString('en-US', {
                          month: 'short',
                          day: 'numeric'
                        })}
                      </span>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </button>
        
        {/* Action buttons - only show when not collapsed and on hover */}
        {!sidebarCollapsed && (
          <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity flex space-x-1">
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleRenameSession(session.id, session.title);
              }}
              className="p-1.5 rounded-lg bg-white shadow-md border border-gray-200 hover:bg-gray-50 transition-colors"
              title="Rename"
            >
              <Edit2 className="h-3 w-3 text-gray-600" />
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDeleteSession(session.id);
              }}
              className="p-1.5 rounded-lg bg-white shadow-md border border-gray-200 hover:bg-red-50 hover:border-red-200 transition-colors"
              title="Delete"
            >
              <Trash2 className="h-3 w-3 text-red-600" />
            </button>
          </div>
        )}
      </div>
    ))}

    {/* Empty state */}
    {chatSessions.length === 0 && !sidebarCollapsed && (
      <div className="text-center py-8 px-4">
        <MessageCircle className="h-12 w-12 text-gray-300 mx-auto mb-3" />
        <p className="text-sm text-gray-500">No conversations yet</p>
        <p className="text-xs text-gray-400">Start a new chat to begin</p>
      </div>
    )}
  </div>

  {/* Sidebar Footer */}
  <div className={`p-3 border-t border-gray-200 space-y-1 ${sidebarCollapsed ? 'px-2' : ''}`}>
    <Link
      href="/dashboard"
      className={`flex items-center space-x-3 p-3 rounded-xl hover:bg-gray-100 transition-colors group ${
        sidebarCollapsed ? 'justify-center' : ''
      }`}
      title="Dashboard"
    >
      <Home className="h-4 w-4 text-gray-500 group-hover:text-gray-700" />
      {!sidebarCollapsed && <span className="text-sm text-gray-700 group-hover:text-gray-900">Dashboard</span>}
    </Link>
    <Link
      href="/courses"
      className={`flex items-center space-x-3 p-3 rounded-xl hover:bg-gray-100 transition-colors group ${
        sidebarCollapsed ? 'justify-center' : ''
      }`}
      title="Courses"
    >
      <BookOpen className="h-4 w-4 text-gray-500 group-hover:text-gray-700" />
      {!sidebarCollapsed && <span className="text-sm text-gray-700 group-hover:text-gray-900">My Courses</span>}
    </Link>
  </div>
</aside>

      {/* Backdrop for mobile */}
      {showHistory && (
        <div
          onClick={() => setShowHistory(false)}
          className="fixed inset-0 bg-black/20 z-40 lg:hidden"
        />
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-screen transition-all duration-300">
        {/* Chat Header */}
        <header className="bg-white border-b px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowHistory(!showHistory)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors lg:hidden"
              >
                <Menu className="h-5 w-5 text-gray-600" />
              </button>
              
              <button
                onClick={() => router.push("/dashboard")}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <ArrowLeft className="h-5 w-5 text-gray-600" />
              </button>
              
              <div>
                <h1 className="text-lg font-semibold text-gray-900">
                  {sessionTitle}
                </h1>
                {courseInfo && (
                  <p className="text-sm text-gray-500 flex items-center space-x-2">
                    <FileText className="h-3 w-3" />
                    <span>{courseInfo.file_count || 0} materials</span>
                    <span className="text-gray-300">•</span>
                    <Target className="h-3 w-3" />
                    <span>{courseInfo.progress || 0}% complete</span>
                  </p>
                )}
              </div>
            </div>
            
            <UserButton />
          </div>
        </header>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto">
            {messages.length === 0 && !isTyping && (
              <div className="py-8 px-4">
                <div className="text-center mb-8">
                  <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-3xl flex items-center justify-center mx-auto mb-4">
                    <Brain className="h-10 w-10 text-white" />
                  </div>
                  <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                    {courseInfo ? `Let's learn ${courseInfo.name}!` : "Ready to learn!"}
                  </h2>
                  <p className="text-gray-600 max-w-md mx-auto">
                    {courseInfo 
                      ? "I have access to your course materials and can help explain concepts, create practice problems, and answer questions."
                      : "Ask me anything about your studies. I'm here to help you learn and understand."}
                  </p>
                </div>

                {/* Suggested Questions */}
                {suggestedQuestions.length > 0 && (
                  <div className="space-y-3 max-w-2xl mx-auto">
                    <p className="text-sm text-gray-500 text-center mb-4">Try asking:</p>
                    {suggestedQuestions.map((question, index) => (
                      <button
                        key={index}
                        onClick={() => handleSendMessage(question)}
                        className="w-full text-left p-4 bg-white rounded-xl border border-gray-200 hover:border-blue-300 hover:shadow-sm transition-all group"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-gray-700 group-hover:text-gray-900">
                            {question}
                          </span>
                          <ChevronRight className="h-4 w-4 text-gray-400 group-hover:text-blue-600 transition-colors" />
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Messages */}
            <div className="px-4 py-6 space-y-6">
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`flex items-start space-x-3 max-w-3xl ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                      message.type === 'user' ? 'bg-blue-600' : 'bg-gray-200'
                    }`}>
                      {message.type === 'user' ? (
                        <User className="h-5 w-5 text-white" />
                      ) : (
                        <Bot className="h-5 w-5 text-gray-600" />
                      )}
                    </div>
                    <div className={`flex-1 ${message.type === 'user' ? 'text-right' : ''}`}>
                      <div className={`inline-block px-4 py-2 rounded-2xl ${
                        message.type === 'user' 
                          ? 'bg-blue-600 text-white' 
                          : 'bg-white border border-gray-200 text-gray-800'
                      }`}>
                        <p className="whitespace-pre-wrap">{message.content}</p>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                </motion.div>
              ))}
              
              {isTyping && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-start space-x-3"
                >
                  <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                    <Bot className="h-5 w-5 text-gray-600" />
                  </div>
                  <div className="bg-white border border-gray-200 rounded-2xl px-4 py-2">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t bg-white">
          <div className="max-w-4xl mx-auto p-4">
            <div className="flex items-end space-x-4">
              <div className="flex-1 relative">
                <textarea
                  ref={textareaRef}
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={courseInfo ? `Ask about ${courseInfo.name}...` : "Ask a question..."}
                  className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows="1"
                  disabled={isStreaming}
                />
                <button
                  onClick={() => handleSendMessage()}
                  disabled={!inputMessage.trim() || isStreaming}
                  className={`absolute right-2 bottom-2 p-2 rounded-lg transition-colors ${
                    inputMessage.trim() && !isStreaming
                      ? 'bg-blue-600 text-white hover:bg-blue-700' 
                      : 'bg-gray-100 text-gray-400'
                  }`}
                >
                  <Send className="h-4 w-4" />
                </button>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-2 text-center">
              AI can make mistakes. Verify important information with your course materials.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}