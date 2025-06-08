// frontend/lib/chatApi.js
/**
 * Chat API service for connecting to FastAPI backend
 */

import { fallbackService } from './fallbackService'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export class ChatApiService {
  constructor() {
    this.baseUrl = API_BASE_URL
    this.isOnline = true
  }

  /**
   * Check if the backend is healthy
   */
  async healthCheck() {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        // Add timeout to prevent hanging
        signal: AbortSignal.timeout(5000)
      })
      
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`)
      }
      
      const data = await response.json()
      this.isOnline = true
      return data
    } catch (error) {
      console.error('Health check failed:', error)
      this.isOnline = false
      throw error
    }
  }

  /**
   * Send a message and get a non-streaming response
   */
  async sendMessage(question) {
    // If offline, use fallback service
    if (!this.isOnline) {
      return fallbackService.generateResponse(question)
    }

    try {
      const response = await fetch(`${this.baseUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question.trim(),
          stream: false
        }),
        // Add timeout
        signal: AbortSignal.timeout(30000)
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(errorData.detail || `API Error: ${response.status}`)
      }

      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error sending message:', error)
      // If network error, mark as offline and use fallback
      if (error.name === 'AbortError' || error.message.includes('fetch')) {
        this.isOnline = false
        return fallbackService.generateResponse(question)
      }
      throw error
    }
  }

  /**
   * Send a message and get a streaming response
   */
  async sendMessageStream(question, onChunk, onMetadata, onError, onComplete) {
    // If offline, simulate streaming with fallback
    if (!this.isOnline) {
      const fallbackResponse = fallbackService.generateResponse(question)
      
      // Simulate metadata
      onMetadata?.({
        sources_count: 0,
        confidence: "Low", 
        avg_score: 0,
        top_score: 0,
        timestamp: new Date().toISOString()
      })

      // Simulate streaming by chunking the response
      const words = fallbackResponse.response.split(' ')
      for (let i = 0; i < words.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 50)) // Simulate typing delay
        onChunk?.(words[i] + (i < words.length - 1 ? ' ' : ''))
      }
      
      onComplete?.()
      return
    }

    try {
      const response = await fetch(`${this.baseUrl}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question.trim(),
          stream: true
        }),
        signal: AbortSignal.timeout(45000)
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(errorData.detail || `API Error: ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      try {
        while (true) {
          const { done, value } = await reader.read()
          
          if (done) {
            onComplete?.()
            break
          }

          const chunk = decoder.decode(value)
          const lines = chunk.split('\n')

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const jsonStr = line.slice(6) // Remove 'data: ' prefix
              
              if (jsonStr.trim() === '') continue

              try {
                const data = JSON.parse(jsonStr)
                
                switch (data.type) {
                  case 'metadata':
                    onMetadata?.(data.data)
                    break
                  case 'content':
                    onChunk?.(data.data)
                    break
                  case 'done':
                    onComplete?.()
                    return
                  case 'error':
                    onError?.(new Error(data.data))
                    return
                  default:
                    console.warn('Unknown data type:', data.type)
                }
              } catch (parseError) {
                console.warn('Failed to parse SSE data:', parseError)
              }
            }
          }
        }
      } finally {
        reader.releaseLock()
      }
    } catch (error) {
      console.error('Error in streaming:', error)
      // If network error, mark as offline and use fallback
      if (error.name === 'AbortError' || error.message.includes('fetch')) {
        this.isOnline = false
        // Use fallback for streaming
        await this.sendMessageStream(question, onChunk, onMetadata, onError, onComplete)
      } else {
        onError?.(error)
      }
    }
  }

  /**
   * Get example questions
   */
  async getExamples() {
    if (!this.isOnline) {
      return fallbackService.getExamples()
    }

    try {
      const response = await fetch(`${this.baseUrl}/examples`, {
        signal: AbortSignal.timeout(5000)
      })
      if (!response.ok) {
        throw new Error(`Failed to get examples: ${response.status}`)
      }
      return await response.json()
    } catch (error) {
      console.error('Error getting examples:', error)
      return fallbackService.getExamples()
    }
  }

  /**
   * Get current online status
   */
  getStatus() {
    return this.isOnline ? 'online' : 'offline'
  }

  /**
   * Force offline mode for testing
   */
  setOfflineMode(offline = true) {
    this.isOnline = !offline
  }
}

// Create singleton instance
export const chatApi = new ChatApiService()