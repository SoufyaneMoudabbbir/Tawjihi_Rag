"use client"

import { useUser } from "@clerk/nextjs"
import { motion } from "framer-motion"
import { useState } from "react"
import { useRouter } from "next/navigation"
import { Plus, Trash2, Eye, Save, ArrowLeft } from "lucide-react"
import DynamicFormRenderer from "@/components/DynamicFormRenderer"

export default function FormBuilderPage() {
  const { user, isLoaded } = useUser()
  const router = useRouter()
  const [formConfig, setFormConfig] = useState({
    title: "New Educational Path Form",
    steps: [
      {
        title: "Personal Information",
        fields: [
          { label: "Full Name", type: "text", name: "fullName", required: true },
          { label: "Age", type: "number", name: "age", required: true },
        ],
      },
    ],
  })
  const [showPreview, setShowPreview] = useState(false)
  const [isSaving, setIsSaving] = useState(false)

  if (!isLoaded) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  const addStep = () => {
    setFormConfig((prev) => ({
      ...prev,
      steps: [
        ...prev.steps,
        {
          title: `Step ${prev.steps.length + 1}`,
          fields: [{ label: "New Field", type: "text", name: `field${Date.now()}`, required: false }],
        },
      ],
    }))
  }

  const removeStep = (stepIndex) => {
    setFormConfig((prev) => ({
      ...prev,
      steps: prev.steps.filter((_, index) => index !== stepIndex),
    }))
  }

  const updateStep = (stepIndex, updates) => {
    setFormConfig((prev) => ({
      ...prev,
      steps: prev.steps.map((step, index) => (index === stepIndex ? { ...step, ...updates } : step)),
    }))
  }

  const addField = (stepIndex) => {
    const newField = {
      label: "New Field",
      type: "text",
      name: `field${Date.now()}`,
      required: false,
    }

    setFormConfig((prev) => ({
      ...prev,
      steps: prev.steps.map((step, index) =>
        index === stepIndex ? { ...step, fields: [...step.fields, newField] } : step,
      ),
    }))
  }

  const removeField = (stepIndex, fieldIndex) => {
    setFormConfig((prev) => ({
      ...prev,
      steps: prev.steps.map((step, index) =>
        index === stepIndex ? { ...step, fields: step.fields.filter((_, fIndex) => fIndex !== fieldIndex) } : step,
      ),
    }))
  }

  const updateField = (stepIndex, fieldIndex, updates) => {
    setFormConfig((prev) => ({
      ...prev,
      steps: prev.steps.map((step, index) =>
        index === stepIndex
          ? {
              ...step,
              fields: step.fields.map((field, fIndex) => (fIndex === fieldIndex ? { ...field, ...updates } : field)),
            }
          : step,
      ),
    }))
  }

  const saveFormConfig = async () => {
    setIsSaving(true)
    try {
      const response = await fetch("/api/form-config", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          userId: user.id,
          config: formConfig,
        }),
      })

      if (response.ok) {
        alert("Form configuration saved successfully!")
      } else {
        alert("Failed to save form configuration")
      }
    } catch (error) {
      console.error("Error saving form config:", error)
      alert("Error saving form configuration")
    } finally {
      setIsSaving(false)
    }
  }

  const fieldTypes = [
    { value: "text", label: "Text" },
    { value: "number", label: "Number" },
    { value: "email", label: "Email" },
    { value: "textarea", label: "Textarea" },
    { value: "select", label: "Select" },
    { value: "checkbox", label: "Checkbox" },
    { value: "radio", label: "Radio" },
  ]

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
              <div>
                <h1 className="text-xl font-semibold text-gray-900">Form Builder</h1>
                <p className="text-sm text-gray-500">Create dynamic questionnaire forms</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowPreview(!showPreview)}
                className="btn-secondary inline-flex items-center space-x-2"
              >
                <Eye className="h-4 w-4" />
                <span>{showPreview ? "Edit" : "Preview"}</span>
              </button>
              <button
                onClick={saveFormConfig}
                disabled={isSaving}
                className="btn-primary inline-flex items-center space-x-2"
              >
                {isSaving ? (
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                ) : (
                  <Save className="h-4 w-4" />
                )}
                <span>Save</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {showPreview ? (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Form Preview</h2>
              <DynamicFormRenderer config={formConfig} />
            </div>
          </motion.div>
        ) : (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">
            {/* Form Title */}
            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">Form Title</label>
              <input
                type="text"
                value={formConfig.title}
                onChange={(e) => setFormConfig((prev) => ({ ...prev, title: e.target.value }))}
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* Steps */}
            {formConfig.steps.map((step, stepIndex) => (
              <div key={stepIndex} className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
                <div className="flex justify-between items-center mb-6">
                  <div className="flex-1 mr-4">
                    <label className="block text-sm font-medium text-gray-700 mb-2">Step Title</label>
                    <input
                      type="text"
                      value={step.title}
                      onChange={(e) => updateStep(stepIndex, { title: e.target.value })}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <button
                    onClick={() => removeStep(stepIndex)}
                    className="p-2 text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                  >
                    <Trash2 className="h-5 w-5" />
                  </button>
                </div>

                {/* Fields */}
                <div className="space-y-4">
                  {step.fields.map((field, fieldIndex) => (
                    <div key={fieldIndex} className="border border-gray-200 rounded-lg p-4">
                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                        <div>
                          <label className="block text-xs font-medium text-gray-700 mb-1">Label</label>
                          <input
                            type="text"
                            value={field.label}
                            onChange={(e) => updateField(stepIndex, fieldIndex, { label: e.target.value })}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-700 mb-1">Type</label>
                          <select
                            value={field.type}
                            onChange={(e) => updateField(stepIndex, fieldIndex, { type: e.target.value })}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                          >
                            {fieldTypes.map((type) => (
                              <option key={type.value} value={type.value}>
                                {type.label}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-700 mb-1">Name</label>
                          <input
                            type="text"
                            value={field.name}
                            onChange={(e) => updateField(stepIndex, fieldIndex, { name: e.target.value })}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                          />
                        </div>
                        <div className="flex items-center justify-between">
                          <label className="flex items-center space-x-2">
                            <input
                              type="checkbox"
                              checked={field.required || false}
                              onChange={(e) => updateField(stepIndex, fieldIndex, { required: e.target.checked })}
                              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <span className="text-xs text-gray-700">Required</span>
                          </label>
                          <button
                            onClick={() => removeField(stepIndex, fieldIndex)}
                            className="p-1 text-red-500 hover:bg-red-50 rounded transition-colors"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </div>

                      {(field.type === "select" || field.type === "radio") && (
                        <div>
                          <label className="block text-xs font-medium text-gray-700 mb-1">
                            Options (comma-separated)
                          </label>
                          <input
                            type="text"
                            value={field.options?.join(", ") || ""}
                            onChange={(e) =>
                              updateField(stepIndex, fieldIndex, {
                                options: e.target.value.split(",").map((opt) => opt.trim()),
                              })
                            }
                            placeholder="Option 1, Option 2, Option 3"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                          />
                        </div>
                      )}
                    </div>
                  ))}

                  <button
                    onClick={() => addField(stepIndex)}
                    className="w-full py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-500 hover:border-blue-300 hover:text-blue-500 transition-colors inline-flex items-center justify-center space-x-2"
                  >
                    <Plus className="h-4 w-4" />
                    <span>Add Field</span>
                  </button>
                </div>
              </div>
            ))}

            {/* Add Step Button */}
            <button
              onClick={addStep}
              className="w-full py-6 border-2 border-dashed border-gray-300 rounded-2xl text-gray-500 hover:border-blue-300 hover:text-blue-500 transition-colors inline-flex items-center justify-center space-x-2"
            >
              <Plus className="h-5 w-5" />
              <span>Add Step</span>
            </button>

            {/* JSON Output */}
            <div className="bg-gray-900 rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">JSON Configuration</h3>
              <pre className="text-green-400 text-sm overflow-x-auto">{JSON.stringify(formConfig, null, 2)}</pre>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}
