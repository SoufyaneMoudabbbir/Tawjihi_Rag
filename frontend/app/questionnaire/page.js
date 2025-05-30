"use client"

import { useUser } from "@clerk/nextjs"
import { motion } from "framer-motion"
import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { ArrowLeft } from "lucide-react"
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
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
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

        // Create new chat session
        const chatResult = await fetch("/api/chats", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            userId: user.id,
            title: `${formData.academicStream || "Educational"} Path - ${new Date().toLocaleDateString()}`,
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
        alert("Failed to submit responses. Please try again.")
      }
    } catch (error) {
      console.error("Error submitting responses:", error)
      alert("An error occurred. Please try again.")
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen py-8">
      {/* Header */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 mb-8">
        <div className="flex items-center justify-between">
          <button
            onClick={() => router.push("/dashboard")}
            className="inline-flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
          >
            <ArrowLeft className="h-5 w-5" />
            <span>Back to Dashboard</span>
          </button>

          {hasCompletedForm && (
            <div className="text-sm text-amber-600 bg-amber-50 px-3 py-2 rounded-lg border border-amber-200">
              You have already completed the questionnaire. This will update your existing responses.
            </div>
          )}
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Title */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-12">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">{formConfig.title}</h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">{formConfig.description}</p>
        </motion.div>

        {/* Questionnaire */}
        <DynamicQuestionnaire config={formConfig} onSubmit={handleSubmit} isSubmitting={isSubmitting} />
      </div>
    </div>
  )
}
