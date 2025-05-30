"use client"

import { motion, AnimatePresence } from "framer-motion"
import { useState, useEffect } from "react"
import { ChevronLeft, ChevronRight, Check, User, GraduationCap, Target, Settings } from "lucide-react"
import FormField from "./FormField"

const iconMap = {
  User,
  GraduationCap,
  Target,
  Settings,
}

export default function DynamicQuestionnaire({ config, onSubmit, isSubmitting }) {
  const [currentStep, setCurrentStep] = useState(0)
  const [formData, setFormData] = useState({})
  const [errors, setErrors] = useState({})
  const [isStepValid, setIsStepValid] = useState(false)

  useEffect(() => {
    validateCurrentStep()
  }, [formData, currentStep])

  const updateFormData = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors((prev) => ({ ...prev, [field]: null }))
    }
  }

  const validateCurrentStep = () => {
    const currentStepData = config.steps[currentStep]
    const stepErrors = {}
    let valid = true

    currentStepData.fields.forEach((field) => {
      const value = formData[field.name]

      // Check if field should be shown (depends on other fields)
      if (field.dependsOn) {
        const dependentValue = formData[field.dependsOn.field]
        if (!field.dependsOn.values.includes(dependentValue)) {
          return // Skip validation for hidden fields
        }
      }

      // Required field validation
      if (field.required) {
        if (!value || (Array.isArray(value) && value.length === 0)) {
          stepErrors[field.name] = `${field.label} is required`
          valid = false
        }
      }

      // Checkbox minimum selections
      if (field.type === "checkbox" && field.minSelections && value) {
        if (value.length < field.minSelections) {
          stepErrors[field.name] = `Please select at least ${field.minSelections} option(s)`
          valid = false
        }
      }

      // Age validation
      if (field.name === "age" && value) {
        const age = Number.parseInt(value)
        if (age < 15 || age > 35) {
          stepErrors[field.name] = "Age must be between 15 and 35"
          valid = false
        }
      }
    })

    setErrors(stepErrors)
    setIsStepValid(valid)
  }

  const handleNext = () => {
    validateCurrentStep()
    if (isStepValid && currentStep < config.steps.length - 1) {
      setCurrentStep(currentStep + 1)
      // Auto-save progress
      saveProgress()
    }
  }

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const saveProgress = async () => {
    // Auto-save to localStorage
    localStorage.setItem(
      "questionnaire-progress",
      JSON.stringify({
        step: currentStep + 1,
        data: formData,
      }),
    )
  }

  const handleSubmit = () => {
    validateCurrentStep()
    if (isStepValid) {
      onSubmit(formData)
    }
  }

  // Load saved progress on mount
  useEffect(() => {
    const saved = localStorage.getItem("questionnaire-progress")
    if (saved) {
      try {
        const { step, data } = JSON.parse(saved)
        setCurrentStep(step || 0)
        setFormData(data || {})
      } catch (error) {
        console.error("Error loading saved progress:", error)
      }
    }
  }, [])

  const currentStepData = config.steps[currentStep]
  const IconComponent = iconMap[currentStepData.icon] || User

  return (
    <div className="space-y-8">
      {/* Enhanced Progress Bar */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          {config.steps.map((step, index) => (
            <div key={step.id} className="flex items-center flex-1">
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: index * 0.1 }}
                className={`
                  relative flex items-center justify-center w-12 h-12 rounded-full border-2 transition-all duration-300
                  ${
                    index <= currentStep
                      ? "bg-blue-600 border-blue-600 text-white shadow-lg"
                      : "bg-white border-gray-300 text-gray-400"
                  }
                `}
              >
                {index < currentStep ? (
                  <Check className="h-5 w-5" />
                ) : (
                  <span className="text-sm font-semibold">{index + 1}</span>
                )}

                {/* Step Icon */}
                {index === currentStep && (
                  <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} className="absolute -top-8">
                    <div className="bg-blue-100 p-2 rounded-full">
                      <IconComponent className="h-4 w-4 text-blue-600" />
                    </div>
                  </motion.div>
                )}
              </motion.div>

              {index < config.steps.length - 1 && (
                <div
                  className={`
                  flex-1 h-1 mx-4 rounded transition-all duration-300
                  ${index < currentStep ? "bg-blue-600" : "bg-gray-200"}
                `}
                />
              )}
            </div>
          ))}
        </div>

        {/* Step Info */}
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <h2 className="text-2xl font-bold text-gray-900 mb-2">{currentStepData.title}</h2>
          <p className="text-gray-600">{currentStepData.description}</p>
        </motion.div>
      </div>

      {/* Form Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
          className="bg-white rounded-2xl shadow-xl border border-gray-100 p-8"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {currentStepData.fields.map((field) => {
              // Handle conditional fields
              if (field.dependsOn) {
                const dependentValue = formData[field.dependsOn.field]
                if (!field.dependsOn.values.includes(dependentValue)) {
                  return null
                }
              }

              return (
                <FormField
                  key={field.name}
                  field={field}
                  value={formData[field.name]}
                  onChange={(value) => updateFormData(field.name, value)}
                  error={errors[field.name]}
                />
              )
            })}
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Navigation */}
      <div className="flex justify-between items-center">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handlePrevious}
          disabled={currentStep === 0}
          className={`
            inline-flex items-center space-x-2 px-6 py-3 rounded-xl font-medium transition-all duration-200
            ${
              currentStep === 0
                ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                : "bg-gray-200 text-gray-700 hover:bg-gray-300 shadow-md hover:shadow-lg"
            }
          `}
        >
          <ChevronLeft className="h-5 w-5" />
          <span>Previous</span>
        </motion.button>

        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-500">
            Step {currentStep + 1} of {config.steps.length}
          </span>
          {!isStepValid && Object.keys(errors).length > 0 && (
            <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-red-500 text-sm">
              • Please complete required fields
            </motion.span>
          )}
        </div>

        {currentStep === config.steps.length - 1 ? (
          <motion.button
            whileHover={{ scale: isStepValid ? 1.02 : 1 }}
            whileTap={{ scale: isStepValid ? 0.98 : 1 }}
            onClick={handleSubmit}
            disabled={!isStepValid || isSubmitting}
            className={`
              inline-flex items-center space-x-2 px-8 py-3 rounded-xl font-medium transition-all duration-200
              ${
                isStepValid && !isSubmitting
                  ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg hover:shadow-xl"
                  : "bg-gray-300 text-gray-500 cursor-not-allowed"
              }
            `}
          >
            {isSubmitting ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                <span>Creating Your Path...</span>
              </>
            ) : (
              <>
                <span>Start My Journey</span>
                <Check className="h-5 w-5" />
              </>
            )}
          </motion.button>
        ) : (
          <motion.button
            whileHover={{ scale: isStepValid ? 1.02 : 1 }}
            whileTap={{ scale: isStepValid ? 0.98 : 1 }}
            onClick={handleNext}
            disabled={!isStepValid}
            className={`
              inline-flex items-center space-x-2 px-6 py-3 rounded-xl font-medium transition-all duration-200
              ${
                isStepValid
                  ? "bg-blue-600 text-white hover:bg-blue-700 shadow-md hover:shadow-lg"
                  : "bg-gray-300 text-gray-500 cursor-not-allowed"
              }
            `}
          >
            <span>Next</span>
            <ChevronRight className="h-5 w-5" />
          </motion.button>
        )}
      </div>
    </div>
  )
}
