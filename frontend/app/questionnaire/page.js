"use client"

import { useUser } from "@clerk/nextjs"
import { motion } from "framer-motion"
import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { ArrowLeft, BookOpen, Brain } from "lucide-react"
import DynamicQuestionnaire from "@/components/forms/DynamicQuestionnaire"
import formConfig from "@/lib/formConfig.json"

export default function QuestionnairePage() {
  const { user, isLoaded } = useUser()
  const router = useRouter()
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [hasCompletedForm, setHasCompletedForm] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (isLoaded && user) {
      checkFormCompletion()
    }
  }, [isLoaded, user])

  const checkFormCompletion = async () => {
    try {
      const response = await fetch(`/api/responses?userId=${user.id}`)
      if (response.ok) {
        setHasCompletedForm(true)
      }
    } catch (error) {
      console.error("Error checking form completion:", error)
    } finally {
      setIsLoading(false)
    }
  }

  if (!isLoaded || isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your learning profile...</p>
        </div>
      </div>
    )
  }

  if (!user) {
    router.push("/")
    return null
  }

  const handleSubmit = async (formData) => {
    setIsSubmitting(true)
    try {
      // Save questionnaire responses
      const responseResult = await fetch("/api/responses", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          userId: user.id,
          responses: formData,
        }),
      })

      if (responseResult.ok) {
        // Clear saved progress
        localStorage.removeItem("questionnaire-progress")

        // Create new chat session with educational focus
        const chatResult = await fetch("/api/chats", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            userId: user.id,
            title: `${formData.majorField || "Academic"} Learning Session - ${new Date().toLocaleDateString()}`,
          }),
        })

        if (chatResult.ok) {
          const chatData = await chatResult.json()
          router.push(`/chat/${chatData.sessionId}`)
        } else {
          router.push("/dashboard")
        }
      } else {
        console.error("Failed to submit responses")
        alert("Failed to submit your learning profile. Please try again.")
      }
    } catch (error) {
      console.error("Error submitting responses:", error)
      alert("An error occurred while setting up your learning profile. Please try again.")
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8">
      {/* Header */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 mb-8">
        <div className="flex items-center justify-between">
          <button
            onClick={() => router.push("/dashboard")}
            className="inline-flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors group"
          >
            <ArrowLeft className="h-5 w-5 group-hover:-translate-x-1 transition-transform" />
            <span>Back to Dashboard</span>
          </button>

          {hasCompletedForm && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-sm text-amber-600 bg-amber-50 px-4 py-2 rounded-xl border border-amber-200 shadow-sm"
            >
              <div className="flex items-center space-x-2">
                <Brain className="h-4 w-4" />
                <span>Updating your learning profile will enhance your personalized experience</span>
              </div>
            </motion.div>
          )}
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Enhanced Title Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }} 
          animate={{ opacity: 1, y: 0 }} 
          className="text-center mb-12"
        >
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl mb-6 shadow-lg">
            <BookOpen className="h-10 w-10 text-white" />
          </div>
          
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            Build Your
            <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent block">
              Learning Profile
            </span>
          </h1>
          
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Help us understand your academic background, learning style, and goals so we can provide 
            personalized educational support tailored just for you.
          </p>

          {/* Benefits Pills */}
          <div className="flex flex-wrap items-center justify-center gap-3 mt-8">
            {[
              "📚 Personalized explanations",
              "🎯 Targeted practice quizzes", 
              "📅 Smart study schedules",
              "📈 Progress tracking"
            ].map((benefit, index) => (
              <motion.div
                key={benefit}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 * index }}
                className="bg-white/80 backdrop-blur-sm px-4 py-2 rounded-full text-sm text-gray-700 border border-gray-200 shadow-sm"
              >
                {benefit}
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Questionnaire */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <DynamicQuestionnaire 
            config={formConfig} 
            onSubmit={handleSubmit} 
            isSubmitting={isSubmitting} 
          />
        </motion.div>

        {/* Footer Info */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="text-center mt-12 mb-8"
        >
          <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 border border-gray-200 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              🔒 Your Privacy Matters
            </h3>
            <p className="text-gray-600 text-sm max-w-2xl mx-auto">
              Your learning profile is used solely to personalize your educational experience. 
              We don't share your information and you can update your preferences anytime.
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
