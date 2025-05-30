"use client"

import { motion } from "framer-motion"
import { useState } from "react"

export default function FormField({ field, value, onChange, error }) {
  const [isFocused, setIsFocused] = useState(false)

  const baseClasses = `
    w-full px-4 py-3 border rounded-xl transition-all duration-200
    focus:ring-2 focus:ring-blue-500 focus:border-transparent
    ${error ? "border-red-300 bg-red-50" : "border-gray-300"}
    ${isFocused ? "shadow-md" : ""}
  `

  const renderField = () => {
    switch (field.type) {
      case "text":
      case "email":
      case "number":
        return (
          <input
            type={field.type}
            value={value || ""}
            onChange={(e) => onChange(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            className={baseClasses}
            placeholder={field.placeholder}
            min={field.min}
            max={field.max}
            required={field.required}
          />
        )

      case "textarea":
        return (
          <textarea
            value={value || ""}
            onChange={(e) => onChange(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            className={baseClasses}
            rows={field.rows || 4}
            placeholder={field.placeholder}
            required={field.required}
          />
        )

      case "select":
        return (
          <select
            value={value || ""}
            onChange={(e) => onChange(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            className={`${baseClasses} bg-white`}
            required={field.required}
          >
            <option value="">{field.placeholder}</option>
            {field.options?.map((option, index) => (
              <option key={index} value={option}>
                {option}
              </option>
            ))}
          </select>
        )

      case "checkbox":
        return (
          <div className="space-y-3">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {field.options?.map((option, index) => (
                <motion.label
                  key={index}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className={`
                    flex items-center space-x-2 p-3 border rounded-lg cursor-pointer transition-colors
                    ${
                      (value || []).includes(option)
                        ? "border-blue-500 bg-blue-50"
                        : "border-gray-200 hover:border-gray-300"
                    }
                  `}
                >
                  <input
                    type="checkbox"
                    checked={(value || []).includes(option)}
                    onChange={(e) => {
                      const currentValues = value || []
                      if (e.target.checked) {
                        // Check max selections limit
                        if (field.maxSelections && currentValues.length >= field.maxSelections) {
                          return
                        }
                        onChange([...currentValues, option])
                      } else {
                        onChange(currentValues.filter((v) => v !== option))
                      }
                    }}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-700 font-medium">{option}</span>
                </motion.label>
              ))}
            </div>
            {field.maxSelections && (
              <p className="text-xs text-gray-500">
                Select up to {field.maxSelections} options ({(value || []).length}/{field.maxSelections})
              </p>
            )}
          </div>
        )

      case "radio":
        return (
          <div className="space-y-3">
            {field.options?.map((option, index) => (
              <motion.label
                key={index}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                className={`
                  flex items-center space-x-3 p-3 border rounded-lg cursor-pointer transition-colors
                  ${value === option ? "border-blue-500 bg-blue-50" : "border-gray-200 hover:border-gray-300"}
                `}
              >
                <input
                  type="radio"
                  name={field.name}
                  value={option}
                  checked={value === option}
                  onChange={(e) => onChange(e.target.value)}
                  className="border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-700 font-medium">{option}</span>
              </motion.label>
            ))}
          </div>
        )

      default:
        return null
    }
  }

  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={field.gridCols || ""}>
      <label className="block text-sm font-medium text-gray-700 mb-2">
        {field.label}
        {field.required && <span className="text-red-500 ml-1">*</span>}
      </label>
      {renderField()}
      {error && (
        <motion.p initial={{ opacity: 0, y: -5 }} animate={{ opacity: 1, y: 0 }} className="text-red-500 text-xs mt-1">
          {error}
        </motion.p>
      )}
    </motion.div>
  )
}
