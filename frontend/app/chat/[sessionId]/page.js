"use client"

import { useUser, UserButton } from "@clerk/nextjs"
import { motion } from "framer-motion"
import { useState, useEffect, useRef } from "react"
import { useRouter, useParams } from "next/navigation"
import { 
  Send, 
  Bot, 
  ArrowLeft, 
  History, 
  BookOpen, 
  FileText, 
  Lightbulb,
  Target,
  Clock,
  Brain
} from "lucide-react"
import ChatMessage from "@/components/ChatMessage"
import ChatHistorySidebar from "@/components/ChatHistorySidebar"

export default function ChatSessionPage() {
  const { user, isLoaded } = useUser()
  const router = useRouter()
  const params = useParams()
  const sessionId = params.sessionId
  const [messages, setMessages] = useState([])
  const [inputMessage, setInputMessage] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const [sessionTitle, setSessionTitle] = useState("")
  const [courseInfo, setCourseInfo] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const [showHistory, setShowHistory] = useState(false)
  const [chatSessions, setChatSessions] = useState([])
  const [suggestedQuestions, setSuggestedQuestions] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    if (isLoaded && user && sessionId) {
      loadChatSession()
      loadChatSessions()
    }
  }, [isLoaded, user, sessionId])

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

  const generateSuggestedQuestions = (course) => {
    const questions = [
      `Explain the key concepts in ${course.name}`,
      `Create a study plan for ${course.name}`,
      `What are common exam questions for this course?`,
      `Help me understand difficult topics in ${course.name}`,
      `Generate practice problems for ${course.name}`
    ]
    setSuggestedQuestions(questions)
  }

  const generateGeneralSuggestedQuestions = () => {
    const questions = [
      "Help me create a study schedule",
      "Explain a concept I'm struggling with",
      "What are effective study techniques?",
      "How can I improve my note-taking?",
      "Generate practice questions for my subject"
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
        
        // Save fallback response
        try {
          await fetch(`/api/chats/${sessionId}/messages`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              message: fallbackResponse,
            }),
          })
        } catch (error) {
          console.error("Error saving fallback response:", error)
        }
      }
    } catch (error) {
      console.error("Error calling RAG API:", error)
      
      // Fallback response for network errors
      const errorResponse = {
        id: Date.now() + 1,
        type: "bot",
        content: "I'm having trouble connecting right now. Please try again in a moment, or feel free to ask your question in a different way.",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorResponse])
    } finally {
      setIsTyping(false)
      setIsStreaming(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleOpenChat = (newSessionId) => {
    router.push(`/chat/${newSessionId}`)
  }

  const handleDeleteSession = async (sessionIdToDelete) => {
    try {
      const response = await fetch(`/api/chats/${sessionIdToDelete}`, {
        method: "DELETE",
      })
      if (response.ok) {
        setChatSessions(chatSessions.filter((session) => session.id !== sessionIdToDelete))
        if (sessionIdToDelete === sessionId) {
          router.push("/dashboard")
        }
      }
    } catch (error) {
      console.error("Error deleting session:", error)
    }
  }

  const handleRenameSession = async (sessionIdToRename, newTitle) => {
    try {
      const response = await fetch(`/api/chats/${sessionIdToRename}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ title: newTitle }),
      })
      if (response.ok) {
        setChatSessions(
          chatSessions.map((session) => (session.id === sessionIdToRename ? { ...session, title: newTitle } : session)),
        )
        if (sessionIdToRename === sessionId) {
          setSessionTitle(newTitle)
        }
      }
    } catch (error) {
      console.error("Error renaming session:", error)
    }
  }

  if (!isLoaded || isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your learning session...</p>
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
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center">
                <Bot className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">{sessionTitle}</h1>
                <div className="flex items-center space-x-2 text-sm text-gray-500">
                  {courseInfo ? (
                    <>
                      <BookOpen className="h-4 w-4" />
                      <span>{courseInfo.name}</span>
                    </>
                  ) : (
                    <>
                      <Brain className="h-4 w-4" />
                      <span>General Learning Session</span>
                    </>
                  )}
                </div>
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

        {/* Chat Container */}
        <div className={`flex-1 transition-all duration-300 ${showHistory ? "ml-80" : ""}`}>
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200 h-[600px] flex flex-col">
              {/* Course Info Banner */}
              {courseInfo && (
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-gray-200 px-6 py-3 rounded-t-2xl">
                  <div className="flex items-center space-x-3">
                    <BookOpen className="h-5 w-5 text-blue-600" />
                    <div>
                      <h3 className="text-sm font-medium text-gray-900">{courseInfo.name}</h3>
                      {courseInfo.professor && (
                        <p className="text-xs text-gray-600">Prof. {courseInfo.professor}</p>
                      )}
                    </div>
                    <div className="ml-auto flex items-center space-x-4 text-xs text-gray-500">
                      <div className="flex items-center space-x-1">
                        <FileText className="h-3 w-3" />
                        <span>{courseInfo.fileCount || 0} materials</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Target className="h-3 w-3" />
                        <span>{courseInfo.progress || 0}% complete</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.length === 0 && !isTyping && (
                  <div className="text-center py-8">
                    <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
                      <Bot className="h-8 w-8 text-white" />
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      {courseInfo ? `Ready to learn ${courseInfo.name}!` : "Ready to start learning!"}
                    </h3>
                    <p className="text-gray-500 mb-6">
                      {courseInfo 
                        ? "I have access to your course materials and can help explain concepts, create practice problems, and answer questions."
                        : "Ask me anything about your studies - I'm here to help you learn and understand new concepts."
                      }
                    </p>
                    
                    {/* Suggested Questions */}
                    {suggestedQuestions.length > 0 && (
                      <div className="max-w-md mx-auto">
                        <p className="text-sm text-gray-600 mb-3">💡 Try asking:</p>
                        <div className="space-y-2">
                          {suggestedQuestions.slice(0, 3).map((question, index) => (
                            <button
                              key={index}
                              onClick={() => handleSendMessage(question)}
                              className="w-full text-left px-4 py-2 text-sm bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg transition-colors"
                            >
                              {question}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {messages.map((message) => (
                  <ChatMessage
                    key={message.id}
                    message={{
                      ...message,
                      timestamp:
                        message.timestamp instanceof Date
                          ? message.timestamp.toLocaleString()
                          : message.timestamp,
                    }}
                  />
                ))}

                {isTyping && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-start space-x-3"
                  >
                    <div className="flex-shrink-0">
                      <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                        <Bot className="h-4 w-4 text-blue-600" />
                      </div>
                    </div>
                    <div className="chat-bubble chat-bubble-bot">
                      <div className="flex items-center space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div
                          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0.1s" }}
                        ></div>
                        <div
                          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0.2s" }}
                        ></div>
                      </div>
                    </div>
                  </motion.div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="border-t border-gray-200 p-4">
                {/* Quick Actions */}
                {messages.length > 0 && suggestedQuestions.length > 0 && (
                  <div className="mb-3">
                    <div className="flex items-center space-x-2 mb-2">
                      <Lightbulb className="h-4 w-4 text-yellow-500" />
                      <span className="text-xs text-gray-600 font-medium">Quick suggestions:</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {[
                        "Explain this concept",
                        "Create practice questions",
                        "Make a study plan",
                        "Quiz me on this topic"
                      ].map((suggestion, index) => (
                        <button
                          key={index}
                          onClick={() => handleSendMessage(suggestion)}
                          disabled={isTyping || isStreaming}
                          className="px-3 py-1 text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full transition-colors disabled:opacity-50"
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                <div className="flex items-end space-x-3">
                  <div className="flex-1">
                    <textarea
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder={
                        courseInfo 
                          ? `Ask about ${courseInfo.name} or request explanations, practice problems, study plans...`
                          : "Ask questions, request explanations, or get help with your studies..."
                      }
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none transition-all duration-200"
                      rows="2"
                      disabled={isTyping || isStreaming}
                    />
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => handleSendMessage()}
                    disabled={!inputMessage.trim() || isTyping || isStreaming}
                    className={`
                      p-3 rounded-xl transition-all duration-200 shadow-md
                      ${
                        (!inputMessage.trim() || isTyping || isStreaming)
                          ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                          : "bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700 hover:shadow-lg"
                      }
                    `}
                  >
                    {isStreaming ? (
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    ) : (
                      <Send className="h-5 w-5" />
                    )}
                  </motion.button>
                </div>

                {/* Status Indicator */}
                {isStreaming && (
                  <div className="flex items-center space-x-2 mt-2 text-xs text-gray-500">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                    <span>Generating response...</span>
                  </div>
                )}
              </div>
            </div>

            {/* Learning Tips */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="mt-6 bg-white/60 backdrop-blur-sm rounded-xl p-4 border border-gray-200"
            >
              <h4 className="text-sm font-medium text-gray-900 mb-2">💡 Learning Tips</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-gray-600">
                <div className="flex items-start space-x-2">
                  <Target className="h-3 w-3 mt-0.5 text-blue-500" />
                  <span>Ask specific questions about concepts you're struggling with</span>
                </div>
                <div className="flex items-start space-x-2">
                  <Clock className="h-3 w-3 mt-0.5 text-green-500" />
                  <span>Request practice problems or quizzes to test your knowledge</span>
                </div>
                <div className="flex items-start space-x-2">
                  <Brain className="h-3 w-3 mt-0.5 text-purple-500" />
                  <span>Ask for study plans or explanations in different formats</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  )
}