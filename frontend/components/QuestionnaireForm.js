"use client"

import { motion } from "framer-motion"

export default function QuestionnaireForm({ step, formData, setFormData }) {
  const updateFormData = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const inputClasses =
    "w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
  const selectClasses =
    "w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors bg-white"

  const renderStep = () => {
    switch (step) {
      case 0: // Personal Information
        return (
          <div className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Full Name</label>
                <input
                  type="text"
                  value={formData.fullName || ""}
                  onChange={(e) => updateFormData("fullName", e.target.value)}
                  className={inputClasses}
                  placeholder="Enter your full name"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Age</label>
                <input
                  type="number"
                  value={formData.age || ""}
                  onChange={(e) => updateFormData("age", e.target.value)}
                  className={inputClasses}
                  placeholder="Your age"
                  min="15"
                  max="35"
                />
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">City</label>
                <select
                  value={formData.city || ""}
                  onChange={(e) => updateFormData("city", e.target.value)}
                  className={selectClasses}
                >
                  <option value="">Select your city</option>
                  <option value="casablanca">Casablanca</option>
                  <option value="rabat">Rabat</option>
                  <option value="marrakech">Marrakech</option>
                  <option value="fes">Fès</option>
                  <option value="tangier">Tangier</option>
                  <option value="agadir">Agadir</option>
                  <option value="meknes">Meknès</option>
                  <option value="oujda">Oujda</option>
                  <option value="kenitra">Kénitra</option>
                  <option value="tetouan">Tétouan</option>
                  <option value="other">Other</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Gender</label>
                <select
                  value={formData.gender || ""}
                  onChange={(e) => updateFormData("gender", e.target.value)}
                  className={selectClasses}
                >
                  <option value="">Select gender</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="prefer-not-to-say">Prefer not to say</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Languages Spoken</label>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {["Arabic", "French", "English", "Spanish", "German", "Other"].map((lang) => (
                  <label key={lang} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={formData.languages?.includes(lang) || false}
                      onChange={(e) => {
                        const languages = formData.languages || []
                        if (e.target.checked) {
                          updateFormData("languages", [...languages, lang])
                        } else {
                          updateFormData(
                            "languages",
                            languages.filter((l) => l !== lang),
                          )
                        }
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">{lang}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        )

      case 1: // Academic Background
        return (
          <div className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Current Education Level</label>
                <select
                  value={formData.educationLevel || ""}
                  onChange={(e) => updateFormData("educationLevel", e.target.value)}
                  className={selectClasses}
                >
                  <option value="">Select level</option>
                  <option value="baccalaureate">Baccalauréat</option>
                  <option value="license">License (Bachelor's)</option>
                  <option value="master">Master's</option>
                  <option value="doctorate">Doctorate</option>
                  <option value="professional">Professional Diploma</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Academic Stream/Field</label>
                <select
                  value={formData.academicStream || ""}
                  onChange={(e) => updateFormData("academicStream", e.target.value)}
                  className={selectClasses}
                >
                  <option value="">Select stream</option>
                  <option value="sciences">Sciences</option>
                  <option value="mathematics">Mathematics</option>
                  <option value="literature">Literature</option>
                  <option value="economics">Economics</option>
                  <option value="technology">Technology</option>
                  <option value="arts">Arts</option>
                  <option value="social-sciences">Social Sciences</option>
                  <option value="engineering">Engineering</option>
                  <option value="medicine">Medicine</option>
                  <option value="business">Business</option>
                </select>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">GPA/Average Grade</label>
                <input
                  type="text"
                  value={formData.gpa || ""}
                  onChange={(e) => updateFormData("gpa", e.target.value)}
                  className={inputClasses}
                  placeholder="e.g., 15.5/20 or 3.5/4.0"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Institution Name</label>
                <input
                  type="text"
                  value={formData.institution || ""}
                  onChange={(e) => updateFormData("institution", e.target.value)}
                  className={inputClasses}
                  placeholder="Current or last institution"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Academic Achievements</label>
              <textarea
                value={formData.achievements || ""}
                onChange={(e) => updateFormData("achievements", e.target.value)}
                className={inputClasses}
                rows="4"
                placeholder="Describe any awards, honors, or notable achievements..."
              />
            </div>
          </div>
        )

      case 2: // Interests & Goals
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Career Interests</label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {[
                  "Technology",
                  "Healthcare",
                  "Education",
                  "Business",
                  "Engineering",
                  "Arts & Design",
                  "Law",
                  "Finance",
                  "Research",
                  "Media",
                  "Tourism",
                  "Agriculture",
                  "Environment",
                  "Social Work",
                  "Other",
                ].map((interest) => (
                  <label key={interest} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={formData.careerInterests?.includes(interest) || false}
                      onChange={(e) => {
                        const interests = formData.careerInterests || []
                        if (e.target.checked) {
                          updateFormData("careerInterests", [...interests, interest])
                        } else {
                          updateFormData(
                            "careerInterests",
                            interests.filter((i) => i !== interest),
                          )
                        }
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">{interest}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Long-term Career Goals</label>
              <textarea
                value={formData.careerGoals || ""}
                onChange={(e) => updateFormData("careerGoals", e.target.value)}
                className={inputClasses}
                rows="4"
                placeholder="Describe your career aspirations and what you hope to achieve..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Preferred Work Environment</label>
              <select
                value={formData.workEnvironment || ""}
                onChange={(e) => updateFormData("workEnvironment", e.target.value)}
                className={selectClasses}
              >
                <option value="">Select preference</option>
                <option value="corporate">Corporate/Office</option>
                <option value="startup">Startup</option>
                <option value="government">Government</option>
                <option value="ngo">NGO/Non-profit</option>
                <option value="freelance">Freelance/Self-employed</option>
                <option value="academic">Academic/Research</option>
                <option value="international">International Organization</option>
              </select>
            </div>
          </div>
        )

      case 3: // Preferences
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Study Abroad Interest</label>
              <div className="space-y-3">
                {[
                  { value: "very-interested", label: "Very interested" },
                  { value: "somewhat-interested", label: "Somewhat interested" },
                  { value: "not-interested", label: "Not interested" },
                  { value: "undecided", label: "Undecided" },
                ].map((option) => (
                  <label key={option.value} className="flex items-center space-x-2">
                    <input
                      type="radio"
                      name="studyAbroad"
                      value={option.value}
                      checked={formData.studyAbroad === option.value}
                      onChange={(e) => updateFormData("studyAbroad", e.target.value)}
                      className="border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">{option.label}</span>
                  </label>
                ))}
              </div>
            </div>

            {formData.studyAbroad !== "not-interested" && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Preferred Study Destinations</label>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {[
                    "France",
                    "Canada",
                    "Germany",
                    "USA",
                    "UK",
                    "Spain",
                    "Belgium",
                    "Netherlands",
                    "Turkey",
                    "UAE",
                    "Other",
                  ].map((country) => (
                    <label key={country} className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={formData.studyDestinations?.includes(country) || false}
                        onChange={(e) => {
                          const destinations = formData.studyDestinations || []
                          if (e.target.checked) {
                            updateFormData("studyDestinations", [...destinations, country])
                          } else {
                            updateFormData(
                              "studyDestinations",
                              destinations.filter((d) => d !== country),
                            )
                          }
                        }}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-700">{country}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Financial Considerations</label>
              <div className="space-y-3">
                {[
                  { value: "scholarship-required", label: "Need scholarship/financial aid" },
                  { value: "family-support", label: "Family can support education costs" },
                  { value: "work-study", label: "Willing to work while studying" },
                  { value: "loan-acceptable", label: "Open to education loans" },
                ].map((option) => (
                  <label key={option.value} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={formData.financialConsiderations?.includes(option.value) || false}
                      onChange={(e) => {
                        const considerations = formData.financialConsiderations || []
                        if (e.target.checked) {
                          updateFormData("financialConsiderations", [...considerations, option.value])
                        } else {
                          updateFormData(
                            "financialConsiderations",
                            considerations.filter((c) => c !== option.value),
                          )
                        }
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">{option.label}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Additional Comments</label>
              <textarea
                value={formData.additionalComments || ""}
                onChange={(e) => updateFormData("additionalComments", e.target.value)}
                className={inputClasses}
                rows="4"
                placeholder="Any additional information you'd like to share..."
              />
            </div>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>
      {renderStep()}
    </motion.div>
  )
}
