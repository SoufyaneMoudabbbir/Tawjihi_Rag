"use client"

import { useUser, UserButton } from "@clerk/nextjs"
import { motion } from "framer-motion"
import { useState, useEffect, useRef } from "react"
import { useRouter, useParams } from "next/navigation"
import { Send, Bot, ArrowLeft, History } from "lucide-react"
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
  const [isLoading, setIsLoading] = useState(true)
  const [showHistory, setShowHistory] = useState(false)
  const [chatSessions, setChatSessions] = useState([])
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
        setSessionTitle(data.title || "Educational Path Exploration")
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

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isTyping) return

    const userMessage = {
      id: Date.now(),
      type: "user",
      content: inputMessage.trim(),
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInputMessage("")
    setIsTyping(true)

    // Save message to database
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

    // Simulate AI response
    setTimeout(async () => {
      const botResponse = {
        id: Date.now() + 1,
        type: "bot",
        content: generateResponse(userMessage.content),
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, botResponse])
      setIsTyping(false)

      // Save bot response to database
      try {
        await fetch(`/api/chats/${sessionId}/messages`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: botResponse,
          }),
        })
      } catch (error) {
        console.error("Error saving bot response:", error)
      }
    }, 1500)
  }

  const generateResponse = (message) => {
    const responses = [
      "Based on your academic background in sciences, I recommend exploring engineering programs at Mohammed V University in Rabat or Hassan II University in Casablanca. Both offer excellent programs with strong industry connections.",
      "For your interest in technology, consider the École Nationale Supérieure d'Informatique et d'Analyse des Systèmes (ENSIAS) or the Institut National des Postes et Télécommunications (INPT). These institutions have excellent placement records.",
      "Given your interest in studying abroad, I suggest looking into the Erasmus+ program partnerships that Moroccan universities have with European institutions. This could be a great pathway to international experience.",
      "For scholarship opportunities, check out the Excellence Scholarship Program by the Ministry of Higher Education, and international scholarships like the Fulbright Program for the US or Campus France for France.",
      "Based on your career goals, I recommend gaining practical experience through internships. Many Moroccan companies offer excellent internship programs that can complement your academic studies.",
    ]
    return responses[Math.floor(Math.random() * responses.length)]
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
              <button
                onClick={() => router.push("/dashboard")}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <ArrowLeft className="h-5 w-5 text-gray-600" />
              </button>
              <Bot className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-xl font-semibold text-gray-900">{sessionTitle}</h1>
                <p className="text-sm text-gray-500">Educational guidance chat</p>
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
            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 h-[600px] flex flex-col">
              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.length === 0 && !isTyping && (
                  <div className="text-center py-12">
                    <Bot className="h-12 w-12 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">Start a conversation to get personalized educational guidance</p>
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

              {/* Input */}
              <div className="border-t border-gray-200 p-4">
                <div className="flex items-end space-x-3">
                  <div className="flex-1">
                    <textarea
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Ask about universities, programs, scholarships, or career advice..."
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none transition-all duration-200"
                      rows="2"
                      disabled={isTyping}
                    />
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleSendMessage}
                    disabled={!inputMessage.trim() || isTyping}
                    className={`
                      p-3 rounded-xl transition-all duration-200 shadow-md
                      ${
                        !inputMessage.trim() || isTyping
                          ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                          : "bg-blue-600 text-white hover:bg-blue-700 hover:shadow-lg"
                      }
                    `}
                  >
                    <Send className="h-5 w-5" />
                  </motion.button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
