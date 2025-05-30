"use client"

import { motion } from "framer-motion"
import { useState } from "react"
import { ChevronLeft, ChevronRight } from "lucide-react"

export default function DynamicFormRenderer({ config }) {
  const [currentStep, setCurrentStep] = useState(0)
  const [formData, setFormData] = useState({})

  const updateFormData = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const handleNext = () => {
    if (currentStep < config.steps.length - 1) {
      setCurrentStep(currentStep + 1)
    }
  }

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const renderField = (field) => {
    const baseClasses =
      "w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"

    switch (field.type) {
      case "text":
      case "email":
      case "number":
        return (
          <input
            type={field.type}
            value={formData[field.name] || ""}
            onChange={(e) => updateFormData(field.name, e.target.value)}
            className={baseClasses}
            placeholder={`Enter ${field.label.toLowerCase()}`}
            required={field.required}
          />
        )

      case "textarea":
        return (
          <textarea
            value={formData[field.name] || ""}
            onChange={(e) => updateFormData(field.name, e.target.value)}
            className={baseClasses}
            rows="4"
            placeholder={`Enter ${field.label.toLowerCase()}`}
            required={field.required}
          />
        )

      case "select":
        return (
          <select
            value={formData[field.name] || ""}
            onChange={(e) => updateFormData(field.name, e.target.value)}
            className={baseClasses}
            required={field.required}
          >
            <option value="">Select {field.label.toLowerCase()}</option>
            {field.options?.map((option, index) => (
              <option key={index} value={option}>
                {option}
              </option>
            ))}
          </select>
        )

      case "checkbox":
        return (
          <div className="space-y-2">
            {field.options?.map((option, index) => (
              <label key={index} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={formData[field.name]?.includes(option) || false}
                  onChange={(e) => {
                    const currentValues = formData[field.name] || []
                    if (e.target.checked) {
                      updateFormData(field.name, [...currentValues, option])
                    } else {
                      updateFormData(
                        field.name,
                        currentValues.filter((v) => v !== option),
                      )
                    }
                  }}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-700">{option}</span>
              </label>
            ))}
          </div>
        )

      case "radio":
        return (
          <div className="space-y-2">
            {field.options?.map((option, index) => (
              <label key={index} className="flex items-center space-x-2">
                <input
                  type="radio"
                  name={field.name}
                  value={option}
                  checked={formData[field.name] === option}
                  onChange={(e) => updateFormData(field.name, e.target.value)}
                  className="border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-700">{option}</span>
              </label>
            ))}
          </div>
        )

      default:
        return null
    }
  }

  const currentStepData = config.steps[currentStep]

  return (
    <div className="space-y-6">
      {/* Progress */}
      <div className="flex items-center justify-between mb-8">
        {config.steps.map((step, index) => (
          <div key={index} className="flex items-center">
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                index <= currentStep ? "bg-blue-600 text-white" : "bg-gray-200 text-gray-500"
              }`}
            >
              {index + 1}
            </div>
            {index < config.steps.length - 1 && (
              <div className={`w-full h-1 mx-4 ${index < currentStep ? "bg-blue-600" : "bg-gray-200"}`} />
            )}
          </div>
        ))}
      </div>

      {/* Step Content */}
      <motion.div
        key={currentStep}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -20 }}
        transition={{ duration: 0.3 }}
      >
        <h3 className="text-xl font-semibold text-gray-900 mb-6">{currentStepData.title}</h3>

        <div className="space-y-6">
          {currentStepData.fields.map((field, index) => (
            <div key={index}>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {field.label}
                {field.required && <span className="text-red-500 ml-1">*</span>}
              </label>
              {renderField(field)}
            </div>
          ))}
        </div>
      </motion.div>

      {/* Navigation */}
      <div className="flex justify-between items-center pt-6">
        <button
          onClick={handlePrevious}
          disabled={currentStep === 0}
          className={`inline-flex items-center space-x-2 px-6 py-3 rounded-xl font-medium transition-colors ${
            currentStep === 0 ? "bg-gray-100 text-gray-400 cursor-not-allowed" : "btn-secondary"
          }`}
        >
          <ChevronLeft className="h-5 w-5" />
          <span>Previous</span>
        </button>

        <span className="text-sm text-gray-500">
          Step {currentStep + 1} of {config.steps.length}
        </span>

        <button
          onClick={handleNext}
          disabled={currentStep === config.steps.length - 1}
          className={`inline-flex items-center space-x-2 px-6 py-3 rounded-xl font-medium transition-colors ${
            currentStep === config.steps.length - 1 ? "bg-gray-100 text-gray-400 cursor-not-allowed" : "btn-primary"
          }`}
        >
          <span>Next</span>
          <ChevronRight className="h-5 w-5" />
        </button>
      </div>

      {/* Form Data Preview */}
      <div className="mt-8 p-4 bg-gray-50 rounded-lg">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Form Data Preview:</h4>
        <pre className="text-xs text-gray-600 overflow-x-auto">{JSON.stringify(formData, null, 2)}</pre>
      </div>
    </div>
  )
}
