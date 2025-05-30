"use client"

import { motion } from "framer-motion"
import { Bot, User } from "lucide-react"

export default function ChatMessage({ message }) {
  const isBot = message.type === "bot"

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex items-start space-x-3 ${isBot ? "" : "flex-row-reverse space-x-reverse"}`}
    >
      <div className="flex-shrink-0">
        <div
          className={`w-8 h-8 rounded-full flex items-center justify-center ${isBot ? "bg-blue-100" : "bg-gray-100"}`}
        >
          {isBot ? <Bot className="h-4 w-4 text-blue-600" /> : <User className="h-4 w-4 text-gray-600" />}
        </div>
      </div>
      <div className={`chat-bubble ${isBot ? "chat-bubble-bot" : "chat-bubble-user"}`}>
        <p className="text-sm leading-relaxed">{message.content}</p>
        <p className={`text-xs mt-2 ${isBot ? "text-gray-500" : "text-blue-200"}`}>
          {message.timestamp}
        </p>
      </div>
    </motion.div>
  )
}
