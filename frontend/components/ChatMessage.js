// frontend/components/ChatMessage.js
"use client"

import { motion } from "framer-motion"
import { Bot, User, AlertCircle, CheckCircle, Info } from "lucide-react"

export default function ChatMessage({ message }) {
  const isUser = message.type === "user"
  const isError = message.isError

  const getConfidenceColor = (confidence) => {
    switch (confidence?.toLowerCase()) {
      case 'high':
        return 'text-green-600 bg-green-50 border-green-200'
      case 'medium':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'low':
        return 'text-red-600 bg-red-50 border-red-200'
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const formatContent = (content) => {
    // Basic markdown-like formatting
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/📚|📖|💡|🎓|🏫|⭐|✅|❌|🔍|📝|💼|🌟/g, (match) => `<span class="text-lg">${match}</span>`)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex items-start space-x-3 ${isUser ? "flex-row-reverse space-x-reverse" : ""}`}
    >
      {/* Avatar */}
      <div className="flex-shrink-0">
        <div className={`
          w-8 h-8 rounded-full flex items-center justify-center
          ${isUser 
            ? "bg-blue-600 text-white" 
            : isError 
              ? "bg-red-100 text-red-600"
              : "bg-blue-100 text-blue-600"
          }
        `}>
          {isUser ? (
            <User className="h-4 w-4" />
          ) : isError ? (
            <AlertCircle className="h-4 w-4" />
          ) : (
            <Bot className="h-4 w-4" />
          )}
        </div>
      </div>

      {/* Message Content */}
      <div className={`max-w-xs lg:max-w-md ${isUser ? "ml-auto" : "mr-auto"}`}>
        <div className={`
          chat-bubble transition-all duration-200
          ${isUser 
            ? "chat-bubble-user" 
            : isError
              ? "bg-red-50 text-red-800 border border-red-200"
              : "chat-bubble-bot"
          }
        `}>
          <div 
            className="whitespace-pre-wrap"
            dangerouslySetInnerHTML={{ __html: formatContent(message.content) }}
          />
        </div>

        {/* Metadata for bot messages */}
        {!isUser && message.metadata && !isError && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="mt-2 flex flex-wrap items-center gap-2 text-xs"
          >
            {/* Confidence Badge */}
            {message.metadata.confidence && (
              <div className={`
                px-2 py-1 rounded-full border text-xs font-medium
                ${getConfidenceColor(message.metadata.confidence)}
              `}>
                <div className="flex items-center space-x-1">
                  {message.metadata.confidence === 'High' ? (
                    <CheckCircle className="h-3 w-3" />
                  ) : (
                    <Info className="h-3 w-3" />
                  )}
                  <span>{message.metadata.confidence} Confidence</span>
                </div>
              </div>
            )}

            {/* Sources Count */}
            {message.metadata.sources_count > 0 && (
              <div className="px-2 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-200 text-xs">
                {message.metadata.sources_count} source{message.metadata.sources_count > 1 ? 's' : ''}
              </div>
            )}

            {/* Score Info (only for high scores) */}
            {message.metadata.top_score > 0.8 && (
              <div className="px-2 py-1 rounded-full bg-green-50 text-green-700 border border-green-200 text-xs">
                High relevance
              </div>
            )}
          </motion.div>
        )}

        {/* Timestamp */}
        <div className={`mt-1 text-xs text-gray-500 ${isUser ? "text-right" : "text-left"}`}>
          {new Date(message.timestamp).toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
          })}
        </div>
      </div>
    </motion.div>
  )
}
