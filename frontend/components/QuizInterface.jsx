"use client"

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Clock, 
  CheckCircle, 
  XCircle, 
  RotateCcw, 
  Trophy,
  Target,
  Brain,
  ArrowRight,
  ArrowLeft
} from 'lucide-react'

export default function QuizInterface({ courseId, chapterId, userId, onComplete, onClose }) {
  const [quiz, setQuiz] = useState(null)
  const [loading, setLoading] = useState(true)
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [userAnswers, setUserAnswers] = useState([])
  const [timeStarted, setTimeStarted] = useState(null)
  const [showResults, setShowResults] = useState(false)
  const [results, setResults] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    loadQuiz()
  }, [courseId, chapterId])

  const loadQuiz = async () => {
    try {
      console.log('🔄 Loading quiz for course', courseId, 'chapter', chapterId)
      
      const response = await fetch(`http://localhost:8000/api/quiz/${courseId}/${chapterId}?user_id=${userId}`)
      
      if (response.ok) {
        const data = await response.json()
        setQuiz(data.quiz_data)
        setUserAnswers(new Array(data.quiz_data.questions.length).fill(null))
        setTimeStarted(Date.now())
        console.log('✅ Quiz loaded:', data.quiz_data.quiz_metadata.title)
      } else {
        console.error('❌ Failed to load quiz')
        alert('Failed to load quiz. Please try again.')
      }
    } catch (error) {
      console.error('❌ Error loading quiz:', error)
      alert('Error loading quiz. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleAnswerSelect = (questionIndex, selectedOption) => {
    const newAnswers = [...userAnswers]
    newAnswers[questionIndex] = selectedOption
    setUserAnswers(newAnswers)
  }

  const submitQuiz = async () => {
    if (userAnswers.includes(null)) {
      alert('Please answer all questions before submitting.')
      return
    }

    setSubmitting(true)
    
    try {
      const timeElapsed = Math.floor((Date.now() - timeStarted) / 1000) // seconds
      
      const response = await fetch('http://localhost:8000/api/quiz/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          quiz_id: quiz.quiz_id, // We'll fix this later
          chapter_id: chapterId,
          course_id: courseId,
          user_answers: userAnswers,
          time_taken: timeElapsed
        })
      })

      if (response.ok) {
        const resultData = await response.json()
        setResults(resultData)
        setShowResults(true)
        console.log('✅ Quiz submitted successfully:', resultData)
      } else {
        alert('Failed to submit quiz. Please try again.')
      }
    } catch (error) {
      console.error('❌ Error submitting quiz:', error)
      alert('Error submitting quiz. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  const retakeQuiz = () => {
    setCurrentQuestion(0)
    setUserAnswers(new Array(quiz.questions.length).fill(null))
    setTimeStarted(Date.now())
    setShowResults(false)
    setResults(null)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-200 border-t-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your quiz...</p>
        </div>
      </div>
    )
  }

  if (!quiz) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-red-50 to-pink-100 flex items-center justify-center">
        <div className="text-center">
          <XCircle className="h-16 w-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Quiz Not Available</h2>
          <p className="text-gray-600 mb-4">Unable to load the quiz for this chapter.</p>
          <button onClick={onClose} className="bg-red-600 text-white px-6 py-2 rounded-lg hover:bg-red-700">
            Go Back
          </button>
        </div>
      </div>
    )
  }

  // Results Screen
  if (showResults) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 flex items-center justify-center p-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white rounded-2xl shadow-2xl p-8 max-w-2xl w-full"
        >
          <div className="text-center mb-8">
            {results.passed ? (
              <Trophy className="h-20 w-20 text-yellow-500 mx-auto mb-4" />
            ) : (
              <Target className="h-20 w-20 text-orange-500 mx-auto mb-4" />
            )}
            
            <h2 className="text-3xl font-bold text-gray-900 mb-2">
              {results.passed ? '🎉 Congratulations!' : '📚 Keep Learning!'}
            </h2>
            
            <div className="text-6xl font-bold mb-4">
              <span className={results.passed ? 'text-green-600' : 'text-orange-600'}>
                {results.score}%
              </span>
            </div>
            
            <p className="text-lg text-gray-600">
              You got {results.correct_answers} out of {results.total_questions} questions correct
            </p>
            
            {results.score > results.previous_best && (
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <p className="text-blue-800 font-medium">
                  🚀 New Personal Best! Previous: {results.previous_best}%
                </p>
              </div>
            )}
          </div>

          {/* Question Review */}
          <div className="space-y-4 mb-8 max-h-64 overflow-y-auto">
            {quiz.questions.map((question, index) => {
              const isCorrect = userAnswers[index] === question.correct
              return (
                <div key={index} className="border rounded-lg p-4 bg-gray-50">
                  <div className="flex items-start space-x-3">
                    {isCorrect ? (
                      <CheckCircle className="h-5 w-5 text-green-500 mt-1 flex-shrink-0" />
                    ) : (
                      <XCircle className="h-5 w-5 text-red-500 mt-1 flex-shrink-0" />
                    )}
                    <div className="flex-1">
                      <p className="font-medium text-gray-900 mb-2">
                        {index + 1}. {question.question}
                      </p>
                      <p className="text-sm text-gray-600">
                        <span className="font-medium">Correct Answer:</span> {question.options[question.correct]}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {question.explanation}
                      </p>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Action Buttons */}
          <div className="flex space-x-4">
            <button
              onClick={retakeQuiz}
              className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-xl hover:bg-blue-700 transition-colors flex items-center justify-center space-x-2"
            >
              <RotateCcw className="h-5 w-5" />
              <span>Retake Quiz</span>
            </button>
            
            <button
              onClick={() => onComplete(results)}
              className="flex-1 bg-gray-600 text-white py-3 px-6 rounded-xl hover:bg-gray-700 transition-colors"
            >
              Continue Learning
            </button>
          </div>
        </motion.div>
      </div>
    )
  }

  // Quiz Taking Screen
  const currentQ = quiz.questions[currentQuestion]
  const progress = ((currentQuestion + 1) / quiz.questions.length) * 100

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-gray-900">{quiz.quiz_metadata.title}</h1>
              <p className="text-sm text-gray-600">
                Question {currentQuestion + 1} of {quiz.questions.length}
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-gray-600">
                <Clock className="h-4 w-4" />
                <span className="text-sm">{quiz.quiz_metadata.estimated_time}</span>
              </div>
              <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
                ✕
              </button>
            </div>
          </div>
          
          {/* Progress Bar */}
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <motion.div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Question Content */}
      <div className="max-w-4xl mx-auto px-6 py-8">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentQuestion}
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -50 }}
            className="bg-white rounded-2xl shadow-lg p-8"
          >
            <div className="mb-8">
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-semibold">
                  {currentQuestion + 1}
                </div>
                <span className="px-3 py-1 bg-blue-100 text-blue-800 text-sm font-medium rounded-full">
                  {currentQ.type === 'multiple_choice' ? 'Multiple Choice' : 'True/False'}
                </span>
              </div>
              
              <h2 className="text-xl font-semibold text-gray-900 leading-relaxed">
                {currentQ.question}
              </h2>
            </div>

            {/* Answer Options */}
            <div className="space-y-3 mb-8">
              {currentQ.options.map((option, index) => (
                <button
                  key={index}
                  onClick={() => handleAnswerSelect(currentQuestion, index)}
                  className={`w-full text-left p-4 rounded-xl border-2 transition-all duration-200 ${
                    userAnswers[currentQuestion] === index
                      ? 'border-blue-500 bg-blue-50 text-blue-900'
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center ${
                      userAnswers[currentQuestion] === index
                        ? 'border-blue-500 bg-blue-500 text-white'
                        : 'border-gray-300'
                    }`}>
                      <span className="text-sm font-medium">
                        {String.fromCharCode(65 + index)}
                      </span>
                    </div>
                    <span className="text-gray-900">{option}</span>
                  </div>
                </button>
              ))}
            </div>

            {/* Navigation */}
            <div className="flex justify-between">
              <button
                onClick={() => setCurrentQuestion(Math.max(0, currentQuestion - 1))}
                disabled={currentQuestion === 0}
                className="flex items-center space-x-2 px-6 py-3 bg-gray-100 text-gray-700 rounded-xl hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <ArrowLeft className="h-4 w-4" />
                <span>Previous</span>
              </button>

              {currentQuestion === quiz.questions.length - 1 ? (
                <button
                  onClick={submitQuiz}
                  disabled={submitting || userAnswers.includes(null)}
                  className="flex items-center space-x-2 px-8 py-3 bg-green-600 text-white rounded-xl hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {submitting ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                      <span>Submitting...</span>
                    </>
                  ) : (
                    <>
                      <Trophy className="h-4 w-4" />
                      <span>Submit Quiz</span>
                    </>
                  )}
                </button>
              ) : (
                <button
                  onClick={() => setCurrentQuestion(Math.min(quiz.questions.length - 1, currentQuestion + 1))}
                  className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors"
                >
                  <span>Next</span>
                  <ArrowRight className="h-4 w-4" />
                </button>
              )}
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}